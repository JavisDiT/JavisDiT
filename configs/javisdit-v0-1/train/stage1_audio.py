# Dataset settings
audio_only=True

dataset = dict(
    type="VariableVideoAudioTextDataset",
    transform_name="resize_crop",
    audio_transform_name="mel_spec_audioldm2",
    audio_only=audio_only
)

# webvid
bucket_config = {  # 5s/it, randomly assigning raw videos to pre-defined and proper buckets
    # image size : {num frame : {accept_probs, batch size}}
    # # 28G？
    # "144p": {51: (1.0, 96), 102: ((1.0, 0.7), 48), 204: ((1.0, 0.3), 24), 408: ((1.0, 0.5), 12)},
    # # 32G
    # "144p": {51: (1.0, 128), 102: ((1.0, 0.7), 64), 204: ((1.0, 0.3), 32), 408: ((1.0, 0.5), 16)},
    # 45G
    "144p": {51: (1.0, 256), 102: ((1.0, 0.7), 128), 204: ((1.0, 0.3), 64), 408: ((1.0, 0.5), 32)},
    # 60-70G
    # "144p": {51: (1.0, 384), 102: ((1.0, 0.7), 192), 204: ((1.0, 0.3), 128), 96: ((1.0, 0.5), 48)},
    # 80G+ 
    # "144p": {51: (1.0, 512), 102: ((1.0, 0.7), 256), 204: ((1.0, 0.3), 128), 408: ((1.0, 0.5), 64)},
}

grad_checkpoint = True

# Acceleration settings
num_workers = 16
num_bucket_build_workers = 8
dtype = "bf16"
plugin = "zero2"

# Model settings
model = dict(
    type="VASTDiT3-XL/2",
    weight_init_from=[
        "./checkpoints/OpenSora-STDiT-v3/model.safetensors"
    ],
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
    # audio generation only
    only_train_audio=True,
    freeze_video_branch=True,
    freeze_y_embedder=False,
    train_st_prior_attn=False,
    train_va_cross_attn=False,
    audio_patch_size=(4, 1)
)
vae = dict(
    type="OpenSoraVAE_V1_2",
    from_pretrained="hpcai-tech/OpenSora-VAE-v1.2",
    micro_frame_size=17,
    micro_batch_size=4,
)
audio_vae = dict(
    type="AudioLDM2",
    from_pretrained="cvssp/audioldm2",
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=300,
)
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    sample_method="logit-normal",
)

# Log settings
seed = 42
outputs = "outputs"
wandb = False
epochs = 50
log_every = 10
ckpt_every = 250
save_total_limit = 2

# optimization settings
load = None
grad_clip = 1.0
lr = 1e-4
ema_decay = 0.99
adam_eps = 1e-15
warmup_steps = 1000

# audio settings
sampling_rate = 16000
mel_bins = 64
audio_cfg = {
    "preprocessing": {
        "audio": {
            "sampling_rate": sampling_rate,
            "max_wav_value": 32768.0,
            "duration": 10.24,
        },
        "stft": {
            "filter_length": 1024,
            "hop_length": 160,
            "win_length": 1024,
        },
        "mel": {
            "n_mel_channels": mel_bins,
            "mel_fmin": 0,
            "mel_fmax": 8000,
        }
    },
    "augmentation": {
        "mixup": 0.0,
    }
}