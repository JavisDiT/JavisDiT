spatial_token_num = 32
temporal_token_num = 32
st_prior_channel = 128

# Dataset settings
dataset = dict(
    type="VariableVideoAudioTextDataset",
    direct_load_video_clip=True,
    transform_name="resize_crop",
    audio_transform_name="mel_spec_audioldm2",
    neg_aug=1,
    neg_aug_kwargs=dict(
        video_augmentation_pool="./data/st_prior/video/SA-V",
        audio_augmentation_pool="./data/st_prior/audio/TAVGBench",
    ),
)
load_text_features = False # TODO: text encoder does not take too much time

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

# Acceleration settings
num_workers = 4
num_bucket_build_workers = 16
dtype = "bf16"
grad_checkpoint = True
plugin = "zero2"

# Model settings
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
model = dict(
    type="STIBPrior",
    imagebind_ckpt_path="./checkpoints",
    spatial_token_num=spatial_token_num,
    temporal_token_num=temporal_token_num,
    out_dim=st_prior_channel,
    hidden_size=512,
    apply_sampling=True,
    encode_va=True,
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True
)

# Log settings
seed = 42
outputs = "outputs"
wandb = False
epochs = 2
log_every = 10
ckpt_every = 200
save_total_limit = 2

# optimization settings
load = None
grad_clip = 1.0
lr = 1e-5
warmup_steps = 100

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