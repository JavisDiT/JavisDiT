import os
import time
from pprint import pformat
import warnings
warnings.filterwarnings('ignore')

import colossalai
import torch
import torch.distributed as dist
from colossalai.cluster import DistCoordinator
from mmengine.runner import set_random_seed
from tqdm import tqdm
import numpy as np
import pandas as pd

from javisdit.acceleration.parallel_states import set_sequence_parallel_group
from javisdit.datasets import save_sample
from javisdit.datasets.aspect import get_image_size, get_num_frames
from javisdit.models.text_encoder.t5 import text_preprocessing
from javisdit.registry import MODELS, SCHEDULERS, build_module
from javisdit.utils.config_utils import parse_configs
from javisdit.utils.inference_utils import (
    add_watermark,
    append_generated,
    append_score_to_prompts,
    apply_va_mask_strategy,
    collect_va_references_batch,
    dframe_to_frame,
    extract_json_from_prompts,
    extract_prompts_loop,
    get_save_path_name,
    load_prompts,
    merge_prompt,
    prepare_multi_resolution_info,
    refine_prompts_by_openai,
    split_prompt
)
from javisdit.utils.misc import all_exists, create_logger, is_distributed, is_main_process, to_torch_dtype


def main():
    torch.set_grad_enabled(False)
    # ======================================================
    # configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs(training=False)
    audio_only = cfg.get('audio_only', False)

    # == device and dtype ==
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dtype = cfg.get("dtype", "fp32")
    assert cfg_dtype in ["fp16", "bf16", "fp32"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # == init distributed env ==
    if is_distributed():
        colossalai.launch_from_torch({})
        coordinator = DistCoordinator()
        enable_sequence_parallelism = coordinator.world_size > 1
        if enable_sequence_parallelism:
            set_sequence_parallel_group(dist.group.WORLD)
    else:
        coordinator = None
        enable_sequence_parallelism = False
    set_random_seed(seed=cfg.get("seed", 1024))

    # == init logger ==
    logger = create_logger()
    logger.info("Inference configuration:\n %s", pformat(cfg.to_dict()))
    verbose = cfg.get("verbose", 1)
    progress_wrap = tqdm if verbose == 1 else (lambda x: x)

    # ======================================================
    # build model & load weights
    # ======================================================
    logger.info("Building models...")
    # == build text-encoder and vae ==
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device, dtype=dtype)
    prior_encoder = build_module(cfg.get('prior_encoder', None), MODELS)
    if prior_encoder is not None:
        prior_encoder = prior_encoder.to(device, dtype).eval()        
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()
    audio_vae = build_module(cfg.audio_vae, MODELS, device=device, dtype=dtype)

    # == prepare video size ==
    image_size = cfg.get("image_size", None)
    if image_size is None:
        resolution = cfg.get("resolution", None)
        aspect_ratio = cfg.get("aspect_ratio", None)
        assert (
            resolution is not None and aspect_ratio is not None
        ), "resolution and aspect_ratio must be provided if image_size is not provided"
        image_size = get_image_size(resolution, aspect_ratio)
    num_frames = get_num_frames(cfg.num_frames)

    # == build diffusion model ==
    input_size = (num_frames, *image_size)
    v_latent_size = vae.get_latent_size(input_size)  # [t//4 for every 17 frame, h//8, w//8]
    model = (
        build_module(
            cfg.model,
            MODELS,
            input_size=v_latent_size,
            in_channels=vae.out_channels,
            caption_channels=text_encoder.output_dim,
            model_max_length=text_encoder.model_max_length,
            enable_sequence_parallelism=enable_sequence_parallelism,
            from_pretrained=cfg.get('model_path', cfg.model.get('from_pretrained')),
            weight_init_from=cfg.model.get('weight_init_from', {}),
        )
        .to(device, dtype)
        .eval()
    )
    text_encoder.y_embedder = model.y_embedder  # HACK: for classifier-free guidance
    if prior_encoder is not None:
        prior_encoder.st_prior_embedder = model.st_prior_embedder

    # == build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # ======================================================
    # inference
    # ======================================================
    # == load prompts ==
    prompts = cfg.get("prompt", None)
    start_idx = cfg.get("start_index", 0)
    if prompts is None:
        if cfg.get("prompt_path", None) is not None:
            prompts = load_prompts(cfg.prompt_path, start_idx, cfg.get("end_index", None),
                                   prompt_key='audio_text' if audio_only else 'text')
        else:
            prompts = [cfg.get("prompt_generator", "")] * 1_000_000  # endless loop

    # == prepare reference ==
    reference_path = cfg.get("reference_path", [("", "")] * len(prompts))
    mask_strategy = cfg.get("mask_strategy", [""] * len(prompts))
    if isinstance(reference_path, str) and os.path.isfile(reference_path):
        reference_df = pd.read_csv(reference_path)
        reference_path = list(zip(reference_df['path'].tolist(), reference_df['audio_path'].tolist()))
    if isinstance(mask_strategy, str):
        mask_strategy = [mask_strategy] * len(prompts)
    assert len(reference_path) == len(prompts), "Length of reference must be the same as prompts"
    assert len(mask_strategy) == len(prompts), "Length of mask_strategy must be the same as prompts"

    # == prepare arguments ==
    fps = cfg.fps
    save_fps = cfg.get("save_fps", fps // cfg.get("frame_interval", 1))
    multi_resolution = cfg.get("multi_resolution", None)
    batch_size = cfg.get("batch_size", 1)
    num_sample = cfg.get("num_sample", 1)
    loop = cfg.get("loop", 1)
    condition_frame_length = cfg.get("condition_frame_length", 5)
    condition_frame_edit = cfg.get("condition_frame_edit", 0.0)
    align = cfg.get("align", None)
    assert loop == 1, "not implemented"
    audio_fps = cfg.get("audio_fps", 16000)

    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)
    sample_name = cfg.get("sample_name", None)
    prompt_as_path = cfg.get("prompt_as_path", False)

    # == Iter over all samples ==
    for i in progress_wrap(range(0, len(prompts), batch_size)):
        # == prepare batch prompts ==
        batch_prompts = prompts[i : i + batch_size]
        ms = mask_strategy[i : i + batch_size]
        refs = reference_path[i : i + batch_size]

        # == get json from prompts ==
        batch_prompts, refs, ms = extract_json_from_prompts(batch_prompts, refs, ms, noimpl=True)
        original_batch_prompts = batch_prompts

        # == get reference for condition ==
        refs = collect_va_references_batch(refs, vae, image_size, 
                                           audio_vae=audio_vae, audio_cfg=cfg.get("audio_cfg", {}))

        # == multi-resolution info ==
        model_args = prepare_multi_resolution_info(
            multi_resolution, len(batch_prompts), image_size, num_frames, fps, device, dtype
        )

        # == Iter over number of sampling for one prompt ==
        for k in range(num_sample):
            # == prepare save paths ==
            save_paths = [
                get_save_path_name(
                    save_dir,
                    sample_name=sample_name,
                    sample_idx=start_idx + idx,
                    prompt=original_batch_prompts[idx],
                    prompt_as_path=prompt_as_path,
                    num_sample=num_sample,
                    k=k,
                )
                for idx in range(len(batch_prompts))
            ]
            if coordinator and coordinator.world_size > 1:
                dist.barrier()
            if all(os.path.exists(f'{_path}.mp4') for _path in save_paths):
                continue

            # NOTE: Skip if the sample already exists
            # This is useful for resuming sampling VBench
            if prompt_as_path and all_exists(save_paths):
                continue

            # == process prompts step by step ==
            # 0. split prompt
            # each element in the list is [prompt_segment_list, loop_idx_list]
            batched_prompt_segment_list = []
            batched_loop_idx_list = []
            for prompt in batch_prompts:
                prompt_segment_list, loop_idx_list = split_prompt(prompt)
                batched_prompt_segment_list.append(prompt_segment_list)
                batched_loop_idx_list.append(loop_idx_list)

            # 1. refine prompt by openai
            if cfg.get("llm_refine", False):
                # only call openai API when
                # 1. seq parallel is not enabled
                # 2. seq parallel is enabled and the process is rank 0
                if not enable_sequence_parallelism or (enable_sequence_parallelism and is_main_process()):
                    for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                        batched_prompt_segment_list[idx] = refine_prompts_by_openai(prompt_segment_list)

                # sync the prompt if using seq parallel
                if enable_sequence_parallelism:
                    coordinator.block_all()
                    prompt_segment_length = [
                        len(prompt_segment_list) for prompt_segment_list in batched_prompt_segment_list
                    ]

                    # flatten the prompt segment list
                    batched_prompt_segment_list = [
                        prompt_segment
                        for prompt_segment_list in batched_prompt_segment_list
                        for prompt_segment in prompt_segment_list
                    ]

                    # create a list of size equal to world size
                    broadcast_obj_list = [batched_prompt_segment_list] * coordinator.world_size
                    dist.broadcast_object_list(broadcast_obj_list, 0)

                    # recover the prompt list
                    batched_prompt_segment_list = []
                    segment_start_idx = 0
                    all_prompts = broadcast_obj_list[0]
                    for num_segment in prompt_segment_length:
                        batched_prompt_segment_list.append(
                            all_prompts[segment_start_idx : segment_start_idx + num_segment]
                        )
                        segment_start_idx += num_segment

            # 2. append score
            for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                batched_prompt_segment_list[idx] = append_score_to_prompts(
                    prompt_segment_list,
                    aes=cfg.get("aes", None),
                    flow=cfg.get("flow", None),
                    camera_motion=cfg.get("camera_motion", None),
                )

            # 3. clean prompt with T5
            for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                batched_prompt_segment_list[idx] = [text_preprocessing(prompt) for prompt in prompt_segment_list]

            # 4. merge to obtain the final prompt
            batch_prompts = []
            for prompt_segment_list, loop_idx_list in zip(batched_prompt_segment_list, batched_loop_idx_list):
                batch_prompts.append(merge_prompt(prompt_segment_list, loop_idx_list))

            # == Iter over loop generation ==
            video_clips, audio_clips = [], []
            for loop_i in range(loop):
                # == get prompt for loop i ==
                batch_prompts_loop = extract_prompts_loop(batch_prompts, loop_i)

                # == add condition frames for loop ==
                if loop_i > 0:
                    raise NotImplementedError
                    refs, ms = append_generated(
                        vae, video_clips[-1], refs, ms, loop_i, condition_frame_length, condition_frame_edit
                    )

                # == sampling ==
                # torch.manual_seed(1024)  ## TODO: fix or not
                vz = torch.randn(len(batch_prompts), vae.out_channels, *v_latent_size, device=device, dtype=dtype)
                audio_length_in_s = num_frames / fps
                az, original_waveform_length = audio_vae.prepare_latents(audio_length_in_s, len(batch_prompts), device=device, dtype=dtype)
                masks = apply_va_mask_strategy(vz, az, refs, ms, loop_i, align=align,
                                               v2a_t_scale=1/5*17/fps/(10.24/1024)/4)

                samples = scheduler.multimodal_sample(
                    model,
                    text_encoder,
                    {'video': vz, 'audio': az},
                    batch_prompts_loop,
                    device=device,
                    additional_args=model_args,
                    progress=verbose >= 2,
                    mask=masks,
                    prior_encoder=prior_encoder,
                )
                video_samples, audio_samples = samples['video'], samples['audio']
                
                video_samples = vae.decode(video_samples.to(dtype), num_frames=num_frames)
                video_clips.append(video_samples)
                audio_samples = audio_vae.decode_audio(audio_samples, original_waveform_length=original_waveform_length)
                audio_clips.append(audio_samples)

            # == save samples ==
            if is_main_process():
                for idx, batch_prompt in enumerate(batch_prompts):
                    if verbose >= 2:
                        logger.info("Prompt: %s", batch_prompt)
                    save_path = save_paths[idx]
                    video = [video_clips[i][idx] for i in range(loop)]
                    audio = [audio_clips[i][idx] for i in range(loop)]
                    for i in range(1, loop):  # TODO: audio
                        raise NotImplementedError
                        video[i] = video[i][:, dframe_to_frame(condition_frame_length) :]
                        # audio[i] = audio[i][:, audio_dframe_to_frame(condition_frame_length) :]
                    video = torch.cat(video, dim=1)
                    audio = torch.cat(audio, dim=0)
                    save_path = save_sample(
                        video,
                        fps=save_fps,
                        audio=audio,
                        audio_fps=audio_fps,
                        save_path=save_path,
                        verbose=verbose >= 2,
                        audio_only=audio_only,
                    )
                    if save_path.endswith(".mp4") and cfg.get("watermark", False):
                        time.sleep(1)  # prevent loading previous generated video
                        add_watermark(save_path)
        start_idx += len(batch_prompts)
    logger.info("Inference finished.")
    logger.info("Saved %s samples to %s", start_idx, save_dir)


if __name__ == "__main__":
    main()
