import os
import random
from contextlib import nullcontext
from datetime import timedelta
from pprint import pformat
from glob import glob
import shutil
import re
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
from javisdit.datasets.dataloader import prepare_dataloader
from javisdit.registry import DATASETS, MODELS, build_module
from javisdit.utils.ckpt_utils import load, save
from javisdit.utils.config_utils import define_experiment_workspace, parse_configs, save_training_config
from javisdit.utils.misc import (
    Timer,
    all_reduce_mean,
    create_logger,
    create_tensorboard_writer,
    format_numel_str,
    get_model_numel,
    to_torch_dtype,
)
from javisdit.utils.train_utils import create_colossalai_plugin


def main():
    # ======================================================
    # 1. configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs(training=True)
    record_time = cfg.get("record_time", False)

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
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=5))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    set_seed(cfg.get("seed", 1024))
    coordinator = DistCoordinator()
    device = get_current_device()

    # == init exp_dir ==
    model_name = None #'prior'
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
            wandb.init(project="minisora", name=exp_name, config=cfg.to_dict(), dir="./outputs/wandb")

    # == init ColossalAI booster ==
    plugin = create_colossalai_plugin(
        plugin=cfg.get("plugin", "zero2"),
        dtype=cfg_dtype,
        grad_clip=cfg.get("grad_clip", 0),
        sp_size=cfg.get("sp_size", 1),
        reduce_bucket_size_in_m=cfg.get("reduce_bucket_size_in_m", 20),
    )
    booster = Booster(plugin=plugin)

    # ======================================================
    # 2. build dataset and dataloader
    # ======================================================
    logger.info("Building dataset...")

    # == load preprocessed data ==
    load_va_features = cfg.get('load_va_features', False)
    save_data = cfg.get('save_data', None)
    if save_data is not None:
        os.makedirs(save_data, exist_ok=True)
    
    # == build dataset ==
    dataset = build_module(cfg.dataset, DATASETS, audio_cfg=cfg.get("audio_cfg"),
                           load_data=cfg.get("load_data"))
    logger.info("Dataset contains %s samples.", len(dataset))

    # == build dataloader ==
    batch_size = cfg.get("batch_size", 1)
    dataloader_args = dict(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=cfg.get("num_workers", 4),
        seed=cfg.get("seed", 1024),
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        process_group=get_data_parallel_group(),
        prefetch_factor=cfg.get("prefetch_factor", None),
    )
    dataloader, sampler = prepare_dataloader(
        bucket_config=cfg.get("bucket_config", None),
        num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
        **dataloader_args,
    )
    total_batch_size = batch_size * dist.get_world_size() // cfg.get("sp_size", 1)
    logger.info("Total batch size: %s", total_batch_size)
    num_steps_per_epoch = len(dataloader)

    # ======================================================
    # 3. build model
    # ======================================================
    logger.info("Building models...")
    
    # == build video vae model ==
    vae = build_module(cfg.get("vae", None), MODELS)
    if vae is not None:
        vae = vae.to(device, dtype).eval()
        vae_out_channels = vae.out_channels
    else:
        vae_out_channels = cfg.get("vae_out_channels", 4)

    # == build audio vae model ==
    audio_vae = build_module(cfg.audio_vae, MODELS, device=device, dtype=dtype)
    if audio_vae is None:
        audio_vae_out_channels = cfg.get('audio_vae_out_channels', 8)
    else:
        audio_vae_out_channels = audio_vae.vae_out_channels
    
    # == build st-prior model ==
    model = build_module(cfg.model, MODELS,
                         video_in_channel=vae_out_channels, 
                         audio_in_channel=audio_vae_out_channels,
                         ).to(device, dtype).train()
    model_numel, model_numel_trainable = get_model_numel(model)
    logger.info(
        "[ST-Prior] Trainable model params: %s, Total model params: %s",
        format_numel_str(model_numel_trainable),
        format_numel_str(model_numel),
    )

    # == setup prior optimizer ==
    optimizer = HybridAdam(
        filter(lambda p: p.requires_grad, model.parameters()),
        adamw_mode=True,
        lr=cfg.get("lr", 1e-5),
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("adam_eps", 1e-8),
    )
    lr_scheduler = None

    # == additional preparation ==
    if cfg.get("grad_checkpoint", False):
        set_grad_checkpoint(model)
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
    start_epoch = start_step = log_step = sampler_start_idx = acc_step = 0
    running_loss = running_spatial_loss = running_temporal_loss = 0.0
    logger.info("Training for %s epochs with %s steps per epoch", cfg_epochs, num_steps_per_epoch)

    # == resume ==
    if cfg.get("load", None) is not None:
        logger.info("Loading checkpoint")
        start_epoch, start_step = load(
            booster,
            cfg.load,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            sampler=sampler,
        )
        dist.barrier()
        logger.info("Loaded checkpoint %s at epoch %s step %s", cfg.load, start_epoch, start_step)

    # =======================================================
    # 5. training loop
    # =======================================================
    dist.barrier()
    timers = {}
    timer_keys = ["load_data", "move_data", "encode", "forward", "backward"]
    for key in timer_keys:
        if record_time:
            timers[key] = Timer(key, coordinator=coordinator)
        else:
            timers[key] = nullcontext()
    for epoch in range(start_epoch, cfg_epochs):
        # == set dataloader to new epoch ==
        sampler.set_epoch(epoch)
        dataiter = iter(dataloader)
        logger.info("Beginning epoch %s...", epoch)

        # == training loop in an epoch ==
        with tqdm(
            enumerate(dataiter, start=start_step),
            desc=f"Epoch {epoch}",
            disable=not coordinator.is_master(),
            total=num_steps_per_epoch,
            initial=start_step,
        ) as pbar:
            for step, batch in pbar:
            # pbar = iter(pbar)
            # while True:
                timer_list = []
                # with timers["load_data"] as load_data_t:
                #     step, batch = next(pbar)
                # timer_list.append(load_data_t)
                
                bs = batch['video'].shape[0]
                neg_num = list(batch['neg_videos'].values())[0].shape[1]
                
                with timers["move_data"] as move_data_t:
                    vx = batch.pop("video").to(device, dtype)  # [B, C, T, H, W]
                    ax = batch.pop("audio").to(device, dtype)  # [B, 1, T, S]
                    # [BxN, C, T, H, W]
                    neg_vx = {aug_type: aug_vx.flatten(0, 1).to(device, dtype) \
                                for aug_type, aug_vx in batch.pop('neg_videos').items()}
                    # [BxN, 1, T, S]
                    neg_ax = {aug_type: aug_ax.flatten(0, 1).to(device, dtype) \
                                for aug_type, aug_ax in batch.pop('neg_audios').items()}
                timer_list.append(move_data_t)

                # # == mixed training setting ==
                # mixed_strategy = cfg.get("mixed_strategy", None)
                # if mixed_strategy == "mixed_video_image":
                #     if random.random() < cfg.get("mixed_image_ratio", 0.0):
                #         x = x[:, :, :1, :, :]
                # elif mixed_strategy == "mixed_video_random":
                #     length = random.randint(1, x.size(2))
                #     x = x[:, :, :length, :, :]

                # == vae encoding ==
                with timers["encode"] as encode_t:
                    if load_va_features:
                        vdims = vx.shape[1:]
                        neg_vx = {aug_type: aug_vx.view(bs, neg_num, *vdims) for \
                                        aug_type, aug_vx in neg_vx.items()}
                        adims = ax.shape[1:]
                        neg_ax = {aug_type: aug_ax.view(bs, neg_num, *adims) for \
                                        aug_type, aug_ax in neg_ax.items()}
                    else:
                        size_list = [vx.shape[0], *[v.shape[0] for v in neg_vx.values()]]
                        with torch.no_grad():
                            for x, neg_x, encode_func in \
                                    [[vx, neg_vx, vae.encode], [ax, neg_ax, audio_vae.encode_audio]]:
                                x = torch.cat([x, *list(neg_x.values())], dim=0)
                                x = encode_func(x)
                                x_list = x.split(size_list, dim=0)
                                dims = x_list[0].shape[1:]
                                for i, aug_type in enumerate(neg_x.keys()):
                                    neg_x[aug_type] = x_list[i+1].view(bs, neg_num, *dims)
                                neg_x['raw'] = x_list[0]
                        vx, ax = neg_vx.pop('raw'), neg_ax.pop('raw')
                timer_list.append(encode_t)

                # == prior extraction & loss calculation ==
                with timers["forward"] as forward_t:
                    text = batch.pop('text')
                    kwargs = {
                        'mode': 'calc_loss', 
                        'video': vx, 'audio': ax, 
                        'neg_videos': neg_vx, 'neg_audios': neg_ax,
                        'frame_width': batch.get('width'),
                        'frame_height': batch.get('height'),
                    }
                    if batch.get('onset', None) is not None:
                        kwargs.update({'onset': batch['onset'].to(device, dtype)})
                    prior_loss, log_dict = model(text, **kwargs)
                timer_list.append(forward_t)

                # == generator backward & update ==
                with timers["backward"] as backward_t:
                    optimizer.zero_grad()
                    booster.backward(loss=prior_loss, optimizer=optimizer)
                    optimizer.step()
                    all_reduce_mean(prior_loss)
                    running_loss += prior_loss.item()
                timer_list.append(backward_t)

                # == update log info ==
                global_step = epoch * num_steps_per_epoch + step
                log_step += 1
                acc_step += 1

                # == logging ==
                if coordinator.is_master() and (global_step + 1) % cfg.get("log_every", 1) == 0:
                    avg_loss = running_loss / log_step
                    # progress bar
                    # pbar.set_postfix({"loss": avg_loss, "step": step, "global_step": global_step})
                    logger.info({"loss": f'{avg_loss:.3f}','step': step, "global_step": global_step, \
                                    **{k: f'{v:.3f}' for k, v in log_dict.items()}})
                    # tensorboard
                    tb_writer.add_scalar("loss", prior_loss.item(), global_step)
                    for k, v in log_dict.items():
                        tb_writer.add_scalar(k, v, global_step)
                    # wandb
                    if cfg.wandb:
                        wandb.log(
                            {
                                "iter": global_step,
                                "num_samples": global_step * total_batch_size,
                                "epoch": epoch,
                                "loss": prior_loss.item(),
                                "avg_loss": avg_loss,
                                **log_dict,
                            },
                            step=global_step,
                        )
                    running_loss = 0.0
                    log_step = 0

                # == checkpoint saving ==
                ckpt_every = cfg.get("ckpt_every", 0)
                if ckpt_every > 0 and (global_step + 1) % ckpt_every == 0:
                    save(
                        booster,
                        exp_dir,
                        model=model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        epoch=epoch,
                        step=step + 1,
                        global_step=global_step + 1,
                        batch_size=cfg.get("batch_size", None),
                        sampler=sampler,
                    )
                    dist.barrier()

                    logger.info(
                        "Saved checkpoint at epoch %s step %s global_step %s to %s",
                        epoch,
                        step + 1,
                        global_step + 1,
                        exp_dir,
                    )

                    if dist.get_rank() == 0:
                        exp_dir_list = glob(os.path.join(exp_dir, 'epoch*-global_step*'))
                        exp_dir_list.sort(key=lambda x: int(re.search(r'global_step(\d+)', x).group(1)) if re.search(r'global_step(\d+)', x) else float('inf'))
                        if save_total_limit is not None and len(exp_dir_list) > save_total_limit:
                            checkpoint = exp_dir_list[0]
                            shutil.rmtree(checkpoint, ignore_errors=True)
                            logger.info(f"{checkpoint} has been deleted successfully as cfg.save_total_limit!")
                    dist.barrier()

                if record_time and dist.get_rank() == 0:
                    log_str = f"Rank {dist.get_rank()} | Epoch {epoch} | Step {step} | "
                    for timer in timer_list:
                        log_str += f"{timer.name}: {timer.elapsed_time:.3f}s | "
                    logger.info(log_str)
                
                if step >= num_steps_per_epoch:
                    break

        sampler.reset()
        start_step = 0


if __name__ == "__main__":
    main()
