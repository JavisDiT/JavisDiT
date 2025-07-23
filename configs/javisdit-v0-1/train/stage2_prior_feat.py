spatial_token_num = 32
temporal_token_num = 32
st_prior_channel = 128

# Dataset settings
dataset = dict(type="BatchFeatureDataset")
load_va_features = True
load_text_features = False # TODO: text encoder does not take too much time

# Acceleration settings
num_workers = 4
num_bucket_build_workers = 16
dtype = "bf16"
grad_checkpoint = True
plugin = "zero2"

# Model settings
# vae = dict(
#     type="OpenSoraVAE_V1_2",
#     from_pretrained="hpcai-tech/OpenSora-VAE-v1.2",
#     micro_frame_size=17,
#     micro_batch_size=4,
# )
# audio_vae = dict(
#     type="AudioLDM2",
#     from_pretrained="cvssp/audioldm2",
# )
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