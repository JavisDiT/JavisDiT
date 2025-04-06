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

audio_only = True

model = dict(
    type="VASTDiT3-XL/2",
    weight_init_from=[],
    from_pretrained="JavisDiT/JavisDiT-v0.1-audio",
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
    # audio generation only
    only_infer_audio=True,
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
    num_sampling_steps=30,
    cfg_scale=7.0,
)

aes = 6.5    # aesthetic score
flow = None  # motion score
