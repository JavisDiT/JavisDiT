import os
from pprint import pformat
from itertools import islice

import colossalai
import torch
import torch.distributed as dist
from tqdm import tqdm

from javisdit.acceleration.parallel_states import get_data_parallel_group, set_data_parallel_group
from javisdit.datasets.dataloader import prepare_dataloader
from javisdit.datasets.datasets import VariableVideoTextDataset
from javisdit.registry import DATASETS, MODELS, build_module
from javisdit.utils.config_utils import parse_configs, save_training_config
from javisdit.utils.misc import FeatureSaver, Timer, create_logger, to_torch_dtype


def main():
    torch.set_grad_enabled(False)
    # ======================================================
    # 1. configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs(training=False)

    # == device and dtype ==
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    cfg_dtype = cfg.get("dtype", "bf16")
    assert cfg_dtype in ["fp16", "bf16"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))

    # == colossalai init distributed training ==
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dtype = cfg.get("dtype", "fp32")
    assert cfg_dtype in ["fp16", "bf16", "fp32"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    colossalai.launch_from_torch({})
    set_data_parallel_group(dist.group.WORLD)

    # == init logger, tensorboard & wandb ==
    logger = create_logger()
    logger.info("Configuration:\n %s", pformat(cfg.to_dict()))

    # ======================================================
    # 2. build dataset and dataloader
    # ======================================================
    # == global variables ==
    bin_size = cfg.bin_size
    save_text_features = cfg.get("save_text_features", False)
    save_compressed_text_features = cfg.get("save_compressed_text_features", False)
    if save_compressed_text_features:
        raise NotImplementedError
    save_text_only = cfg.get("save_text_only", False)

    # resume from a specific batch index
    start_index = cfg.get("start_index", 0)
    end_index = cfg.get("end_index")
    last_micro_batch_access_index = start_index * bin_size
    start_step = 0

    # create save directory
    assert cfg.get("save_dir", None) is not None, "Please specify the save_dir in the config file."
    save_dir = os.path.join(cfg.save_dir, f"s{start_index}_e{end_index}")
    os.makedirs(save_dir, exist_ok=True)
    save_training_config(cfg.to_dict(), save_dir)
    logger.info("Saving features to %s", save_dir)

    saver = FeatureSaver(save_dir, bin_size, start_bin=start_index)
    start_step = saver.get_num_saved()
    if start_step > 0:
        logger.info(f'Found existing data. Start from step {start_step}.')
        last_micro_batch_access_index += start_step

    logger.info("Building dataset...")
    # == build dataset ==
    dataset = build_module(cfg.dataset, DATASETS, audio_cfg=cfg.get("audio_cfg"))
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
        prefetch_factor=1,
        sampler_kwargs={}
    )
    assert isinstance(dataset, VariableVideoTextDataset)
    dataloader_args['sampler_kwargs'] = {'last_micro_batch_access_index': last_micro_batch_access_index}

    dataloader, _ = prepare_dataloader(
        bucket_config=cfg.get("bucket_config", None),
        num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
        **dataloader_args,
    )
    num_steps_per_epoch = len(dataloader)
    # dataloader.batch_sampler.load_state_dict({"last_micro_batch_access_index": start_index})

    # == number of bins ==
    num_bin = num_steps_per_epoch // bin_size
    logger.info("Number of batches: %s", num_steps_per_epoch)
    logger.info("Bin size: %s", bin_size)
    logger.info("Number of bins: %s", num_bin)
    num_bin_to_process = min(num_bin, end_index) - start_index
    logger.info("Start index: %s", start_index)
    logger.info("End index: %s", end_index)
    logger.info("Number of batches to process: %s", num_bin_to_process)

    # ======================================================
    # 3. build model
    # ======================================================
    logger.info("Building models...")
    # == build text-encoder and vae ==
    text_encoder = build_module(cfg.get('text_encoder', None), MODELS, device=device, dtype=dtype)
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()
    audio_vae = build_module(cfg.audio_vae, MODELS, device=device, dtype=dtype)

    # =======================================================
    # 5. training loop
    # =======================================================
    # == training loop in an epoch ==
    dataloader_iter = iter(dataloader)
    log_time = cfg.get("log_time", False)
    total_steps = num_bin_to_process * bin_size
    for _ in tqdm(range(start_step, total_steps), initial=start_step, total=total_steps):
        with Timer("step", log=log_time):
            with Timer("data loading", log=log_time):
                batch = next(dataloader_iter)
                neg_vx, neg_ax = None, None
                if not save_text_only:
                    vx = batch.pop("video").to(device, dtype)  # [B, C, T, H, W]
                    ax = batch.pop("audio").to(device, dtype) # [B, C, T, M]
                    if dataset.neg_aug:
                        neg_vx = {aug_type: aug_vx.flatten(0, 1).to(device, dtype) \
                                for aug_type, aug_vx in batch.pop('neg_videos').items()}
                        neg_ax = {aug_type: aug_ax.flatten(0, 1).to(device, dtype) \
                                for aug_type, aug_ax in batch.pop('neg_audios').items()}
                else:
                    vx, ax = None, None
                y = batch.get("text")
                raw_text = batch.get('raw_text', batch.get("text"))
                batch_num_frames = batch['num_frames']
                batch_fps = batch['fps']
                batch_duration = batch_num_frames / batch_fps
                assert len(torch.unique(batch_duration)) == 1, 'variable durations temporally unsupported'
            
            if not save_text_only:
                with Timer("vae", log=log_time):
                    if neg_vx is not None:
                        size_list = [vx.shape[0], *[v.shape[0] for v in neg_vx.values()]]
                        bs, neg_num = vx.shape[0], dataset.neg_aug
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
                    else:
                        vx = vae.encode(vx)
                        ax = audio_vae.encode_audio(ax)  # [B, C, T, M]
                with Timer("feature to cpu", log=log_time):
                    vx = vx.cpu()
                    ax = ax.cpu()
                    if dataset.neg_aug:
                        neg_vx = {k: v.cpu() for k, v in neg_vx.items()}
                        neg_ax = {k: v.cpu() for k, v in neg_ax.items()}

            batch_dict = {
                "index": batch["index"],
                "x": vx,
                "ax": ax,
                "text": y,
                "raw_text": raw_text,
                "fps": batch["fps"].to(dtype),
                "audio_fps": batch["audio_fps"].to(dtype),
                "height": batch["height"].to(dtype),
                "width": batch["width"].to(dtype),
                "num_frames": batch["num_frames"].to(dtype),
            }
            if dataset.neg_aug:
                batch_dict.update({
                    "neg_vx": neg_vx,
                    "neg_ax": neg_ax
                })
            if dataset.require_onset:
                batch_dict.update({
                    "onset": batch["onset"].to(dtype),
                })

            if save_text_features:
                with Timer("text", log=log_time):
                    text_infos = text_encoder.encode(y)
                    y_feat = text_infos["y"]
                    y_mask = text_infos["mask"]
                    # if save_compressed_text_features:
                    #     y_feat, y_mask = model.encode_text(y_feat, y_mask)
                    #     y_mask = torch.tensor(y_mask)
                with Timer("feature to cpu", log=log_time):
                    y_feat = y_feat.cpu()
                    y_mask = y_mask.cpu()
                batch_dict.update({
                    "y": y_feat, "mask": y_mask,
                })

            saver.update(batch_dict)
    saver.save()


if __name__ == "__main__":
    main()
