resolution = "240p"
aspect_ratio = "9:16"
num_frames = "4s"
fps = 24
audio_fps = 16000
frame_interval = 1
save_fps = 24

save_dir = "./samples/samples/"
seed = 42
batch_size = 1
multi_resolution = "OpenSora"
dtype = "bf16"
loop = 1  # loop for video extension
condition_frame_length = 5  # used for video extension conditioning
align = 5  # TODO: unknown mechanism, maybe for conditional frame alignment?
verbose = 2

spatial_token_num = 32
temporal_token_num = 32
st_prior_channel = 128

model = dict(
    type="VASTDiT3-XL/2",
    weight_init_from=[],
    from_pretrained="JavisDiT/JavisDiT-v0.1-jav",
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
    # video-audio joint generation
    freeze_y_embedder=True,
    freeze_video_branch=True,
    freeze_audio_branch=True,
    train_st_prior_attn=True,
    train_va_cross_attn=True,
    spatial_prior_len=spatial_token_num,
    temporal_prior_len=temporal_token_num,
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
)
prior_encoder = dict(
    type="STIBPrior",
    imagebind_ckpt_path="./checkpoints",
    from_pretrained="JavisDiT/JavisDiT-v0.1-prior",
    spatial_token_num=spatial_token_num,
    temporal_token_num=temporal_token_num,
    out_dim=st_prior_channel,
    hidden_size=512,
    apply_sampling=True,
    encode_va=False,
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
)
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    num_sampling_steps=30,
    cfg_scale=7.0,
)

aes = 6.5    # aesthetic score
flow = None  # motion score
