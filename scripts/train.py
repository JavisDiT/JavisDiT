import os
from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
from pprint import pformat
from glob import glob
import re
import shutil
import pdb
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.distributed as dist
import wandb
import colossalai
from colossalai.booster import Booster
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device, set_seed
from tqdm import tqdm

from javisdit.acceleration.checkpoint import set_grad_checkpoint
from javisdit.acceleration.parallel_states import get_data_parallel_group
from javisdit.datasets.datasets import VariableVideoTextDataset
from javisdit.datasets.dataloader import prepare_dataloader
from javisdit.registry import DATASETS, MODELS, SCHEDULERS, build_module
from javisdit.utils.ckpt_utils import load, model_gathering, model_sharding, record_model_param_shape, save
from javisdit.utils.config_utils import define_experiment_workspace, parse_configs, save_training_config
from javisdit.utils.lr_scheduler import LinearWarmupLR
from javisdit.utils.misc import (
    Timer,
    all_reduce_mean,
    create_logger,
    create_tensorboard_writer,
    format_numel_str,
    get_model_numel,
    requires_grad,
    to_torch_dtype,
    check_exist_pickle,
)
from javisdit.utils.train_utils import VAMaskGenerator, create_colossalai_plugin, update_ema


def main():
    # ======================================================
    # 1. configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs(training=True)
    record_time = cfg.get("record_time", False)
    start_from_scratch = cfg.get("start_from_scratch", False)

    # == device and dtype ==
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    cfg_dtype = cfg.get("dtype", "bf16")
    assert cfg_dtype in ["fp16", "bf16"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))

    # == colossalai init distributed training ==
    # NOTE: A very large timeout is set to avoid some processes exit early
    if cfg.get('host'):
        colossalai.launch_from_openmpi(cfg.host, cfg.port)
    else:
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=5))  # hours=24
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    set_seed(cfg.get("seed", 1024))
    coordinator = DistCoordinator()
    device = get_current_device()

    # == init exp_dir ==
    model_name = cfg.model["type"].replace("/", "-")
    exp_name, exp_dir = define_experiment_workspace(cfg, model_name=model_name)
    coordinator.block_all()
    if coordinator.is_master():
        os.makedirs(exp_dir, exist_ok=True)
        save_training_config(cfg.to_dict(), exp_dir)
    coordinator.block_all()
    save_total_limit = cfg.get("save_total_limit", None)

    # == init logger, tensorboard & wandb ==
    logger = create_logger(exp_dir)
    logger.info("Experiment directory created at %s", exp_dir)
    logger.info("Training configuration:\n %s", pformat(cfg.to_dict()))
    if coordinator.is_master():
        tb_writer = create_tensorboard_writer(exp_dir)
        if cfg.get("wandb", False):
            wandb.init(project="Open-Sora", name=exp_name, config=cfg.to_dict(), dir="./outputs/wandb")

    # == init ColossalAI booster ==
    plugin = create_colossalai_plugin(
        plugin=cfg.get("plugin", "zero2"),
        dtype=cfg_dtype,
        grad_clip=cfg.get("grad_clip", 0),
        sp_size=cfg.get("sp_size", 1),
        reduce_bucket_size_in_m=cfg.get("reduce_bucket_size_in_m", 20),
    )
    booster = Booster(plugin=plugin)
    torch.set_num_threads(1)

    # ======================================================
    # 2. build dataset and dataloader
    # ======================================================
    logger.info("Building dataset...")
    
    # == load preprocessed data ==
    load_va_features = cfg.get('load_va_features', False)
    audio_only = cfg.get('audio_only', False)
    
    # == build dataset ==
    dataset = build_module(cfg.dataset, DATASETS, audio_cfg=cfg.get("audio_cfg"), audio_only=audio_only,
                           load_data=cfg.get("load_data"))
    logger.info("Dataset contains %s samples.", len(dataset))

    # == build dataloader ==
    dataloader_args = dict(
        dataset=dataset,
        batch_size=cfg.get("batch_size", None),
        num_workers=cfg.get("num_workers", 4),
        seed=cfg.get("seed", 1024),
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        process_group=get_data_parallel_group(),
        prefetch_factor=cfg.get("prefetch_factor", None),
    )
    if cfg.get("load", None) is not None and isinstance(dataset, VariableVideoTextDataset) and not start_from_scratch:
        sampler_dict = torch.load(os.path.join(cfg.load, "sampler"))
        last_micro_batch_access_index = sampler_dict['last_micro_batch_access_index']
        dataloader_args['sampler_kwargs'] = {'last_micro_batch_access_index': last_micro_batch_access_index}
    dataloader, sampler = prepare_dataloader(
        bucket_config=cfg.get("bucket_config", None),
        num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
        **dataloader_args,
    )
    num_steps_per_epoch = len(dataloader)

    # ======================================================
    # 3. build model
    # ======================================================
    logger.info("Building models...")

    # == build text-encoder ==
    text_encoder = build_module(cfg.get("text_encoder", None), MODELS, device=device, dtype=dtype)
    if text_encoder is not None:
        text_encoder_output_dim = text_encoder.output_dim
        text_encoder_model_max_length = text_encoder.model_max_length
    else:
        text_encoder_output_dim = cfg.get("text_encoder_output_dim", 4096)
        text_encoder_model_max_length = cfg.get("text_encoder_model_max_length", 300)
    
    # == build prior-encoder ==
    prior_encoder = build_module(cfg.get('prior_encoder', None), MODELS)
    if prior_encoder is not None:
        prior_encoder = prior_encoder.to(device, dtype).eval()    
    
    # == build video vae ==
    vae = build_module(cfg.get("vae", None), MODELS)
    if vae is not None:
        vae = vae.to(device, dtype).eval()
        if getattr(dataset, "num_frames", None) is not None:
            input_size = (dataset.num_frames, *dataset.image_size)
            latent_size = vae.get_latent_size(input_size)
        else:
            latent_size = (None, None, None)
        vae_out_channels = vae.out_channels
    else:
        latent_size = (None, None, None)
        vae_out_channels = cfg.get("vae_out_channels", 4)

    # == build audio vae ==
    audio_vae = build_module(cfg.audio_vae, MODELS, device=device, dtype=dtype)
    if audio_vae is None:
        audio_vae_out_channels = cfg.get('audio_vae_out_channels', 8)
    else:
        audio_vae_out_channels = audio_vae.vae_out_channels

    # == build javisdit diffusion model ==
    model = (
        build_module(
            cfg.model,
            MODELS,
            input_size=latent_size,
            in_channels=vae_out_channels,
            audio_in_channels=audio_vae_out_channels,
            caption_channels=text_encoder_output_dim,
            model_max_length=text_encoder_model_max_length,
            enable_sequence_parallelism=cfg.get("sp_size", 1) > 1,
        )
        .to(device, dtype)
        .train()
    )

    model_numel, model_numel_trainable = get_model_numel(model)
    logger.info(
        "Trainable model params: %s, Total model params: %s",
        format_numel_str(model_numel_trainable),
        format_numel_str(model_numel),
    )

    # == build ema for model ==
    ema = deepcopy(model).to(torch.float32).to(device)
    requires_grad(ema, False)
    ema_shape_dict = record_model_param_shape(ema)
    ema.eval()
    update_ema(ema, model, decay=0, sharded=False)

    # == setup loss function, build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # == setup optimizer ==
    optimizer = HybridAdam(
        filter(lambda p: p.requires_grad, model.parameters()),
        adamw_mode=True,
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("adam_eps", 1e-8),
    )
    warmup_steps = cfg.get("warmup_steps", None)
    if warmup_steps is None:
        lr_scheduler = None
    else:
        lr_scheduler = LinearWarmupLR(optimizer, warmup_steps=cfg.get("warmup_steps"))

    # == additional preparation ==
    if cfg.get("grad_checkpoint", False):
        set_grad_checkpoint(model)
    if cfg.get("mask_ratios", None) is not None:
        mask_generator = VAMaskGenerator(cfg.mask_ratios)
    if load_va_features:
        for m in [vae, audio_vae]:
            if m is None:
                del m
    torch.cuda.empty_cache()

    # =======================================================
    # 4. distributed training preparation with colossalai
    # =======================================================
    logger.info("Preparing for distributed training...")
    # == boosting ==
    # NOTE: we set dtype first to make initialization of model consistent with the dtype; then reset it to the fp32 as we make diffusion scheduler in fp32
    torch.set_default_dtype(dtype)
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloader=dataloader,
    )
    torch.set_default_dtype(torch.float)
    logger.info("Boosting model for distributed training")

    # == global variables ==
    cfg_epochs = cfg.get("epochs", 1000)
    start_epoch = start_step = log_step = acc_step = 0
    running_loss = 0.0
    logger.info("Training for %s epochs with %s steps per epoch", cfg_epochs, num_steps_per_epoch)

    # == resume ==
    if cfg.get("load", None) is not None:
        logger.info("Loading checkpoint")
        ret = load(
            booster,
            cfg.load,
            model=model,
            ema=ema,
            optimizer=optimizer,
            lr_scheduler=None if start_from_scratch else lr_scheduler,
            sampler=None if start_from_scratch else sampler,
        )
        if not start_from_scratch:
            start_epoch, start_step = ret
        logger.info("Loaded checkpoint %s at epoch %s step %s", cfg.load, start_epoch, start_step)

    model_sharding(ema)

    # =======================================================
    # 5. training loop
    # =======================================================
    dist.barrier()
    timers = {}
    timer_keys = [
        "move_data", "encode", "mask", "diffusion", "backward", "update_ema", "reduce_loss",
    ]
    for key in timer_keys:
        if record_time:
            timers[key] = Timer(key, coordinator=coordinator)
        else:
            timers[key] = nullcontext()
    for epoch in range(start_epoch, cfg_epochs):
        # == set dataloader to new epoch ==
        sampler.set_epoch(epoch)
        dataloader_iter = iter(dataloader)
        logger.info("Beginning epoch %s...", epoch)

        # == training loop in an epoch ==
        with tqdm(
            enumerate(dataloader_iter, start=start_step),
            desc=f"Epoch {epoch}",
            disable=not coordinator.is_master(),
            initial=start_step,
            total=num_steps_per_epoch,
        ) as pbar:
            for step, batch in pbar:
                timer_list = []
                with timers["move_data"] as move_data_t:
                    x = batch.pop("video").to(device, dtype)  # [B, C, T, H, W]
                    ax = batch.pop("audio").to(device, dtype) # [B, 1, T, M]
                    batch_num_frames = batch['num_frames']
                    batch_fps = batch['fps']
                    batch_duration = batch_num_frames / batch_fps
                    assert len(torch.unique(batch_duration)) == 1, 'variable durations temporally unsupported'
                    y, raw_text = batch.get("text"), batch.get('raw_text', batch.get("text"))
                if record_time:
                    timer_list.append(move_data_t)

                # == visual and text encoding ==
                with timers["encode"] as encode_t:
                    with torch.no_grad():
                        # Prepare visual and audio inputs
                        if audio_only:  # fake x
                            x = x.repeat(1, vae_out_channels, 1, 1, 1)
                        if load_va_features:
                            x = x.to(device, dtype)
                            ax = ax.to(device, dtype)
                        else:
                            if not audio_only:
                                x = vae.encode(x)  # [B, C, T, H/P, W/P]
                            ax = audio_vae.encode_audio(ax)  # [B, C, T, M]
                        # Prepare text inputs
                        if cfg.get("load_text_features", False):
                            model_args = {"y": y.to(device, dtype)}
                            mask = batch.pop("mask")
                            if isinstance(mask, torch.Tensor):
                                mask = mask.to(device, dtype)
                            model_args["mask"] = mask
                        else:
                            model_args = text_encoder.encode(y)
                        # Prepare spatio-temporal prior
                        if prior_encoder is not None:
                            model_args.update(prior_encoder.encode(raw_text))
                if record_time:
                    timer_list.append(encode_t)

                # == temporal mask ==
                with timers["mask"] as mask_t:
                    mask, ax_mask = None, None
                    if cfg.get("mask_ratios", None) is not None:
                        mask, ax_mask = mask_generator.get_masks(x, ax)  # shape(B, T)
                        model_args["x_mask"] = mask  
                        model_args["ax_mask"] = ax_mask
                if record_time:
                    timer_list.append(mask_t)

                # == video meta info ==
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        model_args[k] = v.to(device, dtype)

                # == diffusion loss computation ==
                with timers["diffusion"] as loss_t:
                    # loss_dict = scheduler.training_losses(model, x, model_args, mask=mask)
                    x = {'video': x, 'audio': ax}
                    model_args.update({'audio_only': audio_only})
                    mask = {'video': mask, 'audio': ax_mask}
                    loss_dict = scheduler.multimodal_training_losses(model, x, model_args, mask=mask)
                if record_time:
                    timer_list.append(loss_t)

                # == backward & update ==
                with timers["backward"] as backward_t:
                    loss = loss_dict["loss"].mean()
                    booster.backward(loss=loss, optimizer=optimizer)
                    optimizer.step()
                    optimizer.zero_grad()

                    # update learning rate
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                if record_time:
                    timer_list.append(backward_t)

                # == update EMA ==
                with timers["update_ema"] as ema_t:
                    update_ema(ema, model.module, optimizer=optimizer, decay=cfg.get("ema_decay", 0.9999))
                if record_time:
                    timer_list.append(ema_t)

                # == update log info ==
                with timers["reduce_loss"] as reduce_loss_t:
                    all_reduce_mean(loss)
                    running_loss += loss.item()
                    global_step = epoch * num_steps_per_epoch + step
                    log_step += 1
                    acc_step += 1
                if record_time:
                    timer_list.append(reduce_loss_t)

                # == logging ==
                if coordinator.is_master() and (global_step + 1) % cfg.get("log_every", 1) == 0:
                    avg_loss = running_loss / log_step
                    # progress bar
                    pbar.set_postfix({"loss": avg_loss, "step": step, "global_step": global_step})
                    logger.info({"loss": avg_loss, "step": step, "global_step": global_step})
                    # tensorboard
                    tb_writer.add_scalar("loss", loss.item(), global_step)
                    # wandb
                    if cfg.get("wandb", False):
                        wandb_dict = {
                            "iter": global_step,
                            "acc_step": acc_step,
                            "epoch": epoch,
                            "loss": loss.item(),
                            "avg_loss": avg_loss,
                            "lr": optimizer.param_groups[0]["lr"],
                        }
                        if record_time:
                            wandb_dict.update(
                                {
                                    "debug/move_data_time": move_data_t.elapsed_time,
                                    "debug/encode_time": encode_t.elapsed_time,
                                    "debug/mask_time": mask_t.elapsed_time,
                                    "debug/diffusion_time": loss_t.elapsed_time,
                                    "debug/backward_time": backward_t.elapsed_time,
                                    "debug/update_ema_time": ema_t.elapsed_time,
                                    "debug/reduce_loss_time": reduce_loss_t.elapsed_time,
                                }
                            )
                        wandb.log(wandb_dict, step=global_step)

                    running_loss = 0.0
                    log_step = 0

                # == checkpoint saving ==
                ckpt_every = cfg.get("ckpt_every", 0)
                if ckpt_every > 0 and (global_step + 1) % ckpt_every == 0:
                    model_gathering(ema, ema_shape_dict)
                    save_dir = save(
                        booster,
                        exp_dir,
                        model=model,
                        ema=ema,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        sampler=sampler,
                        epoch=epoch,
                        step=step + 1,
                        global_step=global_step + 1,
                        batch_size=cfg.get("batch_size", None),
                    )
                    dist.barrier()
                    if dist.get_rank() == 0:
                        model_sharding(ema)

                        logger.info(
                            "Saved checkpoint at epoch %s, step %s, global_step %s to %s",
                            epoch,
                            step + 1,
                            global_step + 1,
                            save_dir,
                        )

                        exp_dir_list = glob(os.path.join(exp_dir, 'epoch*-global_step*'))
                        exp_dir_list.sort(key=lambda x: int(re.search(r'global_step(\d+)', x).group(1)) if re.search(r'global_step(\d+)', x) else float('inf'))
                        if save_total_limit is not None and len(exp_dir_list) > save_total_limit:
                            checkpoint = exp_dir_list[0]
                            shutil.rmtree(checkpoint, ignore_errors=True)
                            logger.info(f"{checkpoint} has been deleted successfully as cfg.save_total_limit!")
                    dist.barrier()

                if record_time:
                    log_str = f"Rank {dist.get_rank()} | Epoch {epoch} | Step {step} | "
                    for timer in timer_list:
                        log_str += f"{timer.name}: {timer.elapsed_time:.3f}s | "
                    logger.info(log_str)

        sampler.reset()
        start_step = 0
        torch.cuda.empty_cache()
    
    model_gathering(ema, ema_shape_dict)
    save_dir = save(
        booster, exp_dir,
        model=model, ema=ema, optimizer=optimizer, lr_scheduler=lr_scheduler, sampler=sampler,
        epoch=epoch, step=step + 1, global_step=global_step + 1, batch_size=cfg.get("batch_size", None),
    )
    logger.info(
        "Saved final checkpoint at epoch %s, step %s, global_step %s to %s",
        epoch, step + 1,  global_step + 1, save_dir,
    )
    dist.barrier()


if __name__ == "__main__":
    main()
