# Dataset settings
dataset = dict(
    type="VariableVideoAudioTextDataset",
    direct_load_video_clip=True,
    transform_name="resize_crop",
    audio_transform_name="mel_spec_audioldm2",
)
load_text_features = False

# webvid
bucket_config = {  # 20s/it, randomly assigning raw videos to pre-defined and proper buckets
    # image size : {num frame : {accept_probs, batch size}}
    "144p": {51: (1.0, 16), 102: ((1.0, 0.5), 12), 204: ((1.0, 0.5), 6), 408: (1.0, 3)},
    # ---
    "256": {51: (0.5, 10), 102: ((0.5, 0.5), 4), 204: ((0.5, 0.5), 2), 408: (1.0, 1)},
    "240p": {51: (0.5, 10), 102: ((0.5, 0.5), 4), 204: ((0.5, 0.5), 2), 408: (1.0, 1)},
    # ---
    "360p": {51: (0.3, 4), 102: ((0.3, 0.5), 2), 204: ((0.3, 0.5), 1)},
    "512": {51: (0.2, 4), 102: ((0.2, 0.5), 2), 204: ((0.2, 0.4), 1)},
    # ---
    "480p": {51: (0.2, 2), 102: ((0.2, 0.5), 1)},
    # ---
    "720p": {51: (0.03, 1)},
    "1024": {51: (0.03, 1)},
}
grad_checkpoint = True

# Acceleration settings
num_workers = 8
num_bucket_build_workers = 16
dtype = "bf16"
plugin = "zero2"

# Model settings
spatial_prior_len = 32
temporal_prior_len = 32
st_prior_channel = 128
model = dict(
    type="VASTDiT3-XL/2",
    weight_init_from=[
        "./checkpoints/JavisDiT-v0.1-audio",
        "./checkpoints/OpenSora-STDiT-v3/model.safetensors",
    ],
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
    # video-audio joint generation
    only_train_audio=False,
    freeze_y_embedder=True,
    freeze_video_branch=True,
    freeze_audio_branch=True,
    train_st_prior_attn=True,
    train_va_cross_attn=True,
    spatial_prior_len=spatial_prior_len,
    temporal_prior_len=temporal_prior_len,
    st_prior_channel=st_prior_channel,
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
    # shardformer=True,
)
prior_encoder = dict(
    type="STIBPrior",
    imagebind_ckpt_path="./checkpoints",
    from_pretrained="JavisDiT/JavisDiT-v0.1-prior",
    spatial_token_num=spatial_prior_len,
    temporal_token_num=temporal_prior_len,
    out_dim=st_prior_channel,
    apply_sampling=True,
    encode_va=False,
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
)
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    sample_method="logit-normal",
)

# Mask settings
# 30%
mask_ratios = {
    "random":              0.01, 
    "video_to_audio":      0.05,   # func1
    "audio_to_video":      0.05,   # func2
    "sound_image_animate": 0.03, 
    "intepolate":          0.03,          
    "quarter_random":      0.005,
    "quarter_head":        0.05,   # func3
    "quarter_tail":        0.005,
    "quarter_head_tail":   0.005,
    "image_random":        0.005,
    "image_head":          0.05,   # func4
    "image_tail":          0.005,
    "image_head_tail":     0.005,
}
# Log settings
seed = 42
outputs = "outputs"
wandb = False
epochs = 2
log_every = 10
ckpt_every = 50
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