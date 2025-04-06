import os
from typing import Tuple, Optional, Literal
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp

from javisdit.acceleration.checkpoint import auto_grad_checkpoint
from javisdit.acceleration.communications import gather_forward_split_backward, split_forward_gather_backward
from javisdit.acceleration.parallel_states import get_sequence_parallel_group
from javisdit.models.layers.blocks import (
    Attention, MultiHeadCrossAttention, BiMultiHeadAttention,
    MMIdentity, Modulator, Additor, SizeEmbedder,
    CaptionEmbedder, PatchEmbed2D, PatchEmbed3D, PositionEmbedding, PositionEmbedding2D,
    T2IFinalLayer, approx_gelu, get_layernorm, t2i_modulate, ada_interpolate1d,
)
from javisdit.registry import MODELS
from javisdit.utils.ckpt_utils import load_checkpoint
from javisdit.utils.misc import get_logger, requires_grad
from javisdit.models.stdit.stdit3 import STDiT3Block, STDiT3, STDiT3Config

logger = get_logger()

class CrossSTDiT3Block(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        mode: Literal["cross_attn", "modulate", "addition"]='cross_attn',
        enable_flash_attn=False,
        enable_layernorm_kernel=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.enable_flash_attn = enable_flash_attn

        self.mode = mode
        if mode == 'cross_attn':
            ca_cls = MultiHeadCrossAttention
        elif mode == 'modulate':
            ca_cls = Modulator
        elif mode == 'addition':
            ca_cls = Additor

        self.norm1s = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.spatial_cross_attn = ca_cls(hidden_size, num_heads)
        self.norm1t = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.temporal_cross_attn = ca_cls(hidden_size, num_heads)
        self.norm2 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(9, hidden_size) / hidden_size**0.5)

    def t_mask_select(self, x_mask, x, masked_x, T, S):
        # x: [B, (T, S), C]
        # mased_x: [B, (T, S), C]
        # x_mask: [B, T]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        masked_x = rearrange(masked_x, "B (T S) C -> B T S C", T=T, S=S)
        x = torch.where(x_mask[:, :, None, None], x, masked_x)
        x = rearrange(x, "B T S C -> B (T S) C")
        return x

    def forward(
        self,
        x,
        spatial_prior,
        temporal_prior,
        t,
        mask=None,  # text mask TODO: assert None
        x_mask=None,  # temporal mask
        t0=None,  # t with timestamp=0
        T=None,  # number of frames
        S=None,  # number of pixel patches
        return_attn_map=False,
    ):
        dtype = x.dtype
        spatial_prior, temporal_prior = spatial_prior.to(dtype), temporal_prior.to(dtype)
        # prepare modulate parameters
        B, N, C = x.shape
        shift_mca_s, scale_mca_s, gate_mca_s, \
        shift_mca_t, scale_mca_t, gate_mca_t, \
        shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.view(B, 9, -1)
        ).chunk(9, dim=1)
        if x_mask is not None:
            shift_mca_s_zero, scale_mca_s_zero, gate_mca_s_zero, \
            shift_mca_t_zero, scale_mca_t_zero, gate_mca_t_zero, \
            shift_mlp_zero, scale_mlp_zero, gate_mlp_zero = (
                self.scale_shift_table[None] + t0.view(B, 9, -1)
            ).chunk(9, dim=1)
        
        # modulate (spatial attention)
        x_m = t2i_modulate(self.norm1s(x), shift_mca_s, scale_mca_s)
        if x_mask is not None:
            x_m_zero = t2i_modulate(self.norm1s(x), shift_mca_s_zero, scale_mca_s_zero)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

        # spatial attention
        x_m = rearrange(x_m, "B (T S) C -> (B T) S C", T=T, S=S)
        # prior: shape(B, S', C) -> shape(B*T, S', C) 
        spatial_prior = spatial_prior.unsqueeze(1).repeat(1, T, 1, 1).flatten(0,1)
        assert spatial_prior.shape[0] == x_m.shape[0] and spatial_prior.shape[-1] == x_m.shape[-1]
        x_m = self.spatial_cross_attn(x_m, spatial_prior, mask, return_attn_map=return_attn_map)
        if return_attn_map:
            x_m, spatial_attn = x_m
        x_m = rearrange(x_m, "(B T) S C -> B (T S) C", T=T, S=S)

        # modulate (spatial attention)
        x_m_s = gate_mca_s * x_m
        if x_mask is not None:
            x_m_s_zero = gate_mca_s_zero * x_m
            x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)

        # residual
        x = x + self.drop_path(x_m_s)
        
        # modulate (temporal attention)
        x_m = t2i_modulate(self.norm1t(x), shift_mca_t, scale_mca_t)
        if x_mask is not None:
            x_m_zero = t2i_modulate(self.norm1t(x), shift_mca_t_zero, scale_mca_t_zero)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

        # temporal attention
        x_m = rearrange(x_m, "B (T S) C -> (B S) T C", T=T, S=S)
        # prior: shape(B, T', C) -> shape(B*S, T', C) 
        temporal_prior = temporal_prior.unsqueeze(1).repeat(1, S, 1, 1).flatten(0,1)
        assert temporal_prior.shape[0] == x_m.shape[0] and temporal_prior.shape[-1] == x_m.shape[-1]
        x_m = self.temporal_cross_attn(x_m, temporal_prior, mask, return_attn_map=return_attn_map)
        if return_attn_map:
            x_m, temporal_attn = x_m
        x_m = rearrange(x_m, "(B S) T C -> B (T S) C", T=T, S=S)

        # modulate (temporal attention)
        x_m_t = gate_mca_t * x_m
        if x_mask is not None:
            x_m_t_zero = gate_mca_t_zero * x_m
            x_m_t = self.t_mask_select(x_mask, x_m_t, x_m_t_zero, T, S)

        # residual
        x = x + self.drop_path(x_m_t)

        # modulate (MLP)
        x_m = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
        if x_mask is not None:
            x_m_zero = t2i_modulate(self.norm2(x), shift_mlp_zero, scale_mlp_zero)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

        # MLP
        x_m = self.mlp(x_m)

        # modulate (MLP)
        x_m_s = gate_mlp * x_m
        if x_mask is not None:
            x_m_s_zero = gate_mlp_zero * x_m
            x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)

        # residual
        x = x + self.drop_path(x_m_s)

        if return_attn_map:
            x = (x, spatial_attn, temporal_attn)

        return x


class CrossModalityBiAttentionBlock(nn.Module):
    def __init__(self, m1_dim, m2_dim, hidden_size, num_heads, 
                 drop_path=0.0, enable_layernorm_kernel=False):
        super().__init__()
        self.m1_dim = m1_dim
        self.m2_dim = m2_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.attn_norm_m1 = get_layernorm(m1_dim, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.attn_norm_m2 = get_layernorm(m2_dim, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.cross_attn = BiMultiHeadAttention(m1_dim, m2_dim, hidden_size, num_heads, dropout=0.0)

        # TODO: MLP?
        self.gamma_m1 = nn.Parameter(torch.zeros((m1_dim)), requires_grad=True)
        self.gamma_m2 = nn.Parameter(torch.zeros((m2_dim)), requires_grad=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(self, xs: Tuple[torch.Tensor], attention_masks: Optional[Tuple[torch.Tensor]]=(None,None)):
        # x1: shape(B, T*S, C), x2: shape(B, R*M, C)
        x1, x2 = xs
        attention_mask_1, attention_mask_2 = attention_masks
        if attention_mask_1 is not None or attention_mask_2 is not None:
            raise NotImplementedError('attention mask is currently unsupported for video-audio cross attention')

        x_m1, x_m2 = self.attn_norm_m1(x1), self.attn_norm_m2(x2)

        dx_m1, dx_m2 = self.cross_attn(x_m1, x_m2, attention_mask_1, attention_mask_2)

        x1 = x1 + self.drop_path(self.gamma_m1 * dx_m1)
        x2 = x2 + self.drop_path(self.gamma_m2 * dx_m2)

        return x1, x2


class VASTDiT3Config(STDiT3Config):
    model_type = "VASTDiT3"

    def __init__(
        self,
        ## audio params
        audio_input_size=(None, None),
        audio_in_channels=8,
        audio_patch_size=(4, 1),
        ## TODO: currently doest not support different hidden_size
        # audio_hidden_size=None,
        ## video branch
        only_train_audio=False,
        only_infer_audio=False,
        freeze_video_branch=True,
        freeze_audio_branch=False,
        ## spatio-temporal prior cross attention
        train_st_prior_attn=True,
        train_va_cross_attn=True,
        spatial_prior_len=32,
        temporal_prior_len=32,
        st_prior_channel=128,
        st_prior_utilize='cross_attn',
        weight_init_from=None,
        require_onset=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.audio_input_size = audio_input_size
        self.audio_in_channels = audio_in_channels
        self.audio_patch_size = audio_patch_size
        self.audio_hidden_size = kwargs.get('hidden_size')

        self.only_train_audio = only_train_audio
        self.only_infer_audio = only_infer_audio
        self.freeze_video_branch = freeze_video_branch
        self.freeze_audio_branch = freeze_audio_branch

        self.train_st_prior_attn = train_st_prior_attn
        self.train_va_cross_attn = train_va_cross_attn
        self.st_prior_channel = st_prior_channel
        self.spatial_prior_len = spatial_prior_len
        self.temporal_prior_len = temporal_prior_len
        self.st_prior_utilize = st_prior_utilize
        self.weight_init_from = weight_init_from
        self.require_onset = require_onset


class VASTDiT3(STDiT3):
    config_class = VASTDiT3Config

    def __init__(self, config: VASTDiT3Config):
        self.config: VASTDiT3Config
        super().__init__(config)
        if config.enable_sequence_parallelism:
            # logger.warning('enable sequence parallelism might cause inferior generation performance')
            raise NotImplementedError('enable sequence parallelism might cause inferior generation performance')
        self.audio_in_channels = config.audio_in_channels
        self.audio_out_channels = config.audio_in_channels * 2 if config.pred_sigma else config.audio_in_channels

        # model size related
        self.audio_hidden_size = config.audio_hidden_size
        self.audio_patch_size = config.audio_patch_size

        # input size related
        self.audio_input_size = config.audio_input_size
        # self.audio_pos_embed = PositionEmbedding2D(config.audio_hidden_size)
        self.audio_pos_embed = PositionEmbedding(config.audio_hidden_size)
        # self.audio_rope = RotaryEmbedding(dim=self.audio_hidden_size // self.num_heads)

        # embedding
        self.ax_embedder = PatchEmbed2D(config.audio_patch_size, config.audio_in_channels, config.audio_hidden_size)
        self.audio_y_embedder = CaptionEmbedder(
            in_channels=config.caption_channels,
            hidden_size=config.audio_hidden_size,
            uncond_prob=config.class_dropout_prob,
            act_layer=approx_gelu,
            token_num=config.model_max_length,
        )
        self.st_t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 9 * config.hidden_size, bias=True),
        )
        self.st_prior_embedder = CaptionEmbedder(
            in_channels=config.st_prior_channel,
            hidden_size=config.hidden_size,
            uncond_prob=config.class_dropout_prob,
            act_layer=approx_gelu,
            token_num=config.spatial_prior_len + config.temporal_prior_len,
        )
        if config.require_onset:
            # self.onset_embedder = LabelEmbedder(2, config.hidden_size, 0.)
            self.onset_embedder = SizeEmbedder(config.hidden_size)

        # spatial blocks
        drop_path = [x.item() for x in torch.linspace(0, self.drop_path, config.depth)]
        self.audio_spatial_blocks = nn.ModuleList(
            [
                STDiT3Block(
                    hidden_size=config.audio_hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    drop_path=drop_path[i],
                    qk_norm=config.qk_norm,
                    enable_flash_attn=config.enable_flash_attn,
                    enable_layernorm_kernel=config.enable_layernorm_kernel,
                    enable_sequence_parallelism=config.enable_sequence_parallelism,
                )
                for i in range(config.depth)
            ]
        )

        # temporal blocks
        drop_path = [x.item() for x in torch.linspace(0, self.drop_path, config.depth)]
        self.audio_temporal_blocks = nn.ModuleList(
            [
                STDiT3Block(
                    hidden_size=config.audio_hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    drop_path=drop_path[i],
                    qk_norm=config.qk_norm,
                    enable_flash_attn=config.enable_flash_attn,
                    enable_layernorm_kernel=config.enable_layernorm_kernel,
                    enable_sequence_parallelism=config.enable_sequence_parallelism,
                    # temporal
                    temporal=True,
                    rope=self.rope.rotate_queries_or_keys,
                )
                for i in range(config.depth)
            ]
        )

        # video spatio-temporal blocks
        drop_path = [x.item() for x in torch.linspace(0, self.drop_path, config.depth)]
        self.video_st_prior_blocks = nn.ModuleList(
            [
                CrossSTDiT3Block(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    drop_path=drop_path[i],
                    mode=config.st_prior_utilize,
                    enable_flash_attn=config.enable_flash_attn,
                    enable_layernorm_kernel=config.enable_layernorm_kernel,
                )
                for i in range(config.depth)
            ]
        )

        # audio spatio-temporal blocks
        drop_path = [x.item() for x in torch.linspace(0, self.drop_path, config.depth)]
        self.audio_st_prior_blocks = nn.ModuleList(
            [
                CrossSTDiT3Block(
                    hidden_size=config.audio_hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    drop_path=drop_path[i],
                    mode=config.st_prior_utilize,
                    enable_flash_attn=config.enable_flash_attn,
                    enable_layernorm_kernel=config.enable_layernorm_kernel,
                )
                for i in range(config.depth)
            ]
        )

        # video-audio cross-attention blocks
        drop_path = [x.item() for x in torch.linspace(0, self.drop_path, config.depth)]
        self.va_cross_blocks = nn.ModuleList(
            [
                CrossModalityBiAttentionBlock(
                    m1_dim=config.hidden_size,
                    m2_dim=config.audio_hidden_size,
                    hidden_size=min(config.hidden_size, config.audio_hidden_size)//2,
                    num_heads=config.num_heads//2,
                    drop_path=drop_path[i],
                    enable_layernorm_kernel=config.enable_layernorm_kernel,
                )
                for i in range(config.depth)
            ]
        )

        # final layer
        self.audio_final_layer = T2IFinalLayer(config.audio_hidden_size, 
                                               np.prod(self.audio_patch_size), 
                                               self.audio_out_channels)

        # initialize
        self.initialize_va_weights()
        # TODO: reuse null y_embedding
        self.audio_y_embedder.y_embedding.data = self.y_embedder.y_embedding.data

        # adjust modules
        if config.only_train_temporal or config.freeze_video_branch or config.freeze_audio_branch:
            # reused for two branches: t_block, fps_embedder t_embedder
            requires_grad(self, False)
            if config.only_train_temporal:
                raise NotImplementedError('Not supported in current paradigm')
                requires_grad(self.temporal_blocks, True)
                requires_grad(self.audio_temporal_blocks, True)
            trainable_blocks = [self.y_embedder, self.audio_y_embedder, 
                                self.st_t_block, self.st_prior_embedder, 
                                self.video_st_prior_blocks, self.audio_st_prior_blocks,
                                self.va_cross_blocks]
            if not config.freeze_video_branch:
                trainable_blocks.extend([self.x_embedder, self.spatial_blocks, self.temporal_blocks, self.final_layer])
            if not config.freeze_audio_branch:
                trainable_blocks.extend([self.ax_embedder, self.audio_spatial_blocks, self.audio_temporal_blocks, self.audio_final_layer])
            for block in trainable_blocks:
                requires_grad(block, True)
        
        if config.freeze_y_embedder:
            requires_grad(self.y_embedder, False)
            requires_grad(self.audio_y_embedder, False)
        
        mm_identities = [MMIdentity() for _ in range(config.depth)]
        if config.only_train_audio or config.only_infer_audio:
            requires_grad(self.y_embedder, False)
            del self.spatial_blocks; self.spatial_blocks = mm_identities
            del self.temporal_blocks; self.temporal_blocks = mm_identities
            requires_grad(self.final_layer, False)
        
        if not config.train_st_prior_attn:
            del self.st_t_block; self.st_t_block = MMIdentity()
            del self.st_prior_embedder; self.st_prior_embedder = MMIdentity()
            del self.audio_st_prior_blocks; self.audio_st_prior_blocks = mm_identities
            del self.video_st_prior_blocks; self.video_st_prior_blocks = mm_identities

        if not config.train_va_cross_attn:
            del self.va_cross_blocks; self.va_cross_blocks = mm_identities

    def initialize_va_weights(self):
        # Initialize newly-introduced blocks
        for block in self.audio_temporal_blocks:
            nn.init.constant_(block.attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.mlp.fc2.weight, 0)
        # Spatio-Temporal Prior Cross-Attention
        for block in self.video_st_prior_blocks:
            nn.init.constant_(block.spatial_cross_attn.proj.weight, 0)
            nn.init.constant_(block.temporal_cross_attn.proj.weight, 0)
            nn.init.constant_(block.mlp.fc2.weight, 0)
        for block in self.audio_st_prior_blocks:
            nn.init.constant_(block.spatial_cross_attn.proj.weight, 0)
            nn.init.constant_(block.temporal_cross_attn.proj.weight, 0)
            nn.init.constant_(block.mlp.fc2.weight, 0)
        
        if self.config.weight_init_from is not None:
            init_paths = self.config.weight_init_from
            if isinstance(init_paths, str):
                init_paths = [init_paths]
            for pretrain_path in init_paths:
                load_checkpoint(self, pretrain_path, verbose=False)
            # copy parameters from video to audio
            if self.config.only_train_audio:
                for v_blocks, a_blocks in zip([self.spatial_blocks, self.temporal_blocks, [self.y_embedder]], 
                                              [self.audio_spatial_blocks, self.audio_temporal_blocks, [self.audio_y_embedder]]):
                    for v_block, a_block in zip(v_blocks, a_blocks):
                        a_block.load_state_dict(deepcopy(v_block.state_dict()))
                logger.info('audio `spatial_blocks`, `temporal_blocks`, `y_embedder` are initialized from video branch.')

    def get_audio_size(self, x):
        B, _, Ts, M = x.size()
        # hard embedding
        assert Ts % self.audio_patch_size[0] == 0
        assert M % self.audio_patch_size[1] == 0
        Ta, Sa = Ts // self.audio_patch_size[0], M // self.audio_patch_size[1]
        return Ta, Sa

    def encode_audio_text(self, y, mask=None):
        y = self.audio_y_embedder(y, self.training)  # [B, 1, N_token, C]
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, self.hidden_size)
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, self.hidden_size)
        return y, y_lens

    def forward(self, x, timestep, y, mask=None, x_mask=None, fps=None, height=None, width=None, **kwargs):
        dtype = self.x_embedder.proj.weight.dtype
        vx, ax = x.pop('video', None), x.pop('audio', None)
        B = vx.size(0)
        vx, ax = vx.to(dtype), ax.to(dtype)
        timestep = timestep.to(dtype)
        y = y.to(dtype)
        spatial_prior = kwargs.pop('spatial_prior', None)
        temporal_prior = kwargs.pop('temporal_prior', None)
        if spatial_prior is not None:
            assert temporal_prior is not None
            assert len(spatial_prior.shape) == len(temporal_prior.shape) == 3  # shape(B,N,C)
            assert spatial_prior.shape[1] == self.config.spatial_prior_len
            assert temporal_prior.shape[1] == self.config.temporal_prior_len
            assert spatial_prior.shape[-1] == temporal_prior.shape[-1] == self.config.st_prior_channel
            st_prior = torch.cat((spatial_prior, temporal_prior), dim=1)
            st_prior = self.st_prior_embedder(st_prior.unsqueeze(1), self.training).squeeze(1)
            spatial_prior, temporal_prior = \
                st_prior.split([spatial_prior.shape[1], temporal_prior.shape[1]], dim=1)
        ax_mask = kwargs.pop('ax_mask', None)
        if ax_mask is not None:
            # TODO: stupid manual stride for ax_embedder
            # ax_mask = ax_mask.view(B, -1, self.config.audio_patch_size[0]).any(dim=-1)
            ax_mask = ax_mask[:, ::self.config.audio_patch_size[0]]  
            assert ax_mask.shape[1] == ax.shape[2] // self.config.audio_patch_size[0]

        # === get pos embed ===
        # video
        B, _, Tx, Hx, Wx = vx.size()
        T, H, W = self.get_dynamic_size(vx)
        # audio
        _, _, Ta, Sa = ax.size()
        R, M = self.get_audio_size(ax)  # T, S

        # adjust for sequence parallelism
        # we need to ensure H * W is divisible by sequence parallel size
        # for simplicity, we can adjust the height to make it divisible
        if self.enable_sequence_parallelism:
            sp_size = dist.get_world_size(get_sequence_parallel_group())
            if H % sp_size != 0:
                h_pad_size = sp_size - H % sp_size
            else:
                h_pad_size = 0
            if R % sp_size != 0:
                r_pad_size = sp_size - R % sp_size
            else:
                r_pad_size = 0

            if h_pad_size > 0:
                hx_pad_size = h_pad_size * self.patch_size[1]

                # pad vx along the H dimension
                H += h_pad_size
                vx = F.pad(vx, (0, 0, 0, hx_pad_size))

            if r_pad_size > 0:
                rx_pad_size = r_pad_size * self.audio_patch_size[0]

                # pad ax along the R(T) dimension
                R += r_pad_size
                ax = F.pad(ax, (0, 0, rx_pad_size, 0))

        S = H * W
        base_size = round(S**0.5)
        resolution_sq = (height[0].item() * width[0].item()) ** 0.5
        scale = resolution_sq / self.input_sq_size
        pos_emb = self.pos_embed(vx, H, W, scale=scale, base_size=base_size)

        # au_pos_emb = self.audio_pos_embed(ax, R, M)  # MelSpectrogram has fixed bins
        au_pos_emb = self.audio_pos_embed(ax, M)  # MelSpectrogram has fixed bins

        # === get timestep embed ===
        t = self.t_embedder(timestep, dtype=vx.dtype)  # [B, C]
        fps = self.fps_embedder(fps.unsqueeze(1), B)
        t = t + fps  ## timestamp 和 fps 的 embedding 直接相加，再过MLP
        t_mlp, t_st_mlp = self.t_block(t), self.st_t_block(t)
        t0 = t0_mlp = t0_st_mlp = None
        if x_mask is not None:
            assert ax_mask is not None
            t0_timestep = torch.zeros_like(timestep)
            t0 = self.t_embedder(t0_timestep, dtype=vx.dtype)
            t0 = t0 + fps
            t0_mlp, t0_st_mlp = self.t_block(t0), self.st_t_block(t0)

        # === get y embed ===
        ## text embedding  TODO: audio text
        if self.config.skip_y_embedder:
            raise NotImplementedError('Unsupported for skipping y and ay embedder')
            y_lens = mask
            if isinstance(y_lens, torch.Tensor):
                y_lens = y_lens.long().tolist()
            ay, ay_lens = y, y_lens
        else:
            ay, ay_lens = self.encode_audio_text(y, mask)
            y, y_lens = self.encode_text(y, mask)

        # === get vx & ax embed ===
        vx = self.x_embedder(vx)  # [B, N, C]
        vx = rearrange(vx, "B (T S) C -> B T S C", T=T, S=S)
        vx = vx + pos_emb
        ax = self.ax_embedder(ax) # [B, N, C]
        ax = rearrange(ax, "B (T S) C -> B T S C", T=R, S=M)
        ax = ax + au_pos_emb

        # === get onset embed ===
        if self.config.require_onset:
            onset_prior = kwargs.pop('onset_prior').transpose(1,2)  # shape(B,1,N)
            vx_onset = ada_interpolate1d(onset_prior, T, force_interpolate=True, mode='linear', align_corners=False)
            ax_onset = ada_interpolate1d(onset_prior, R, force_interpolate=True, mode='linear', align_corners=False)
            vx = vx + self.onset_embedder(vx_onset.squeeze(1), B).view(B, T, 1, self.hidden_size)
            ax = ax + self.onset_embedder(ax_onset.squeeze(1), B).view(B, R, 1, self.hidden_size)

        # shard over the sequence dim if sp is enabled
        if self.enable_sequence_parallelism:
            vx = split_forward_gather_backward(vx, get_sequence_parallel_group(), dim=2, grad_scale="down")
            S = S // dist.get_world_size(get_sequence_parallel_group())
            ax = split_forward_gather_backward(ax, get_sequence_parallel_group(), dim=2, grad_scale="down")
            M = M // dist.get_world_size(get_sequence_parallel_group())

        vx = rearrange(vx, "B T S C -> B (T S) C", T=T, S=S)
        ax = rearrange(ax, "B T S C -> B (T S) C", T=R, S=M)

        # === blocks ===
        return_attn_map = kwargs.get('return_attn_map', False)
        def _parse_attn_map(attn_map: torch.Tensor):
            attn_map = attn_map[:attn_map.shape[0]//2]  # CFG
            attn_map = attn_map.softmax(dim=-2).mean(dim=(1,3))  # average #head and #prior
            return attn_map.detach().cpu().to(torch.float16)
        attn_maps = {}
        for i, (v_s_blk, v_t_blk, v_p_st_blk, a_s_blk, a_t_blk, a_p_st_blk, va_st_blk) in \
                enumerate(zip(self.spatial_blocks, self.temporal_blocks, self.video_st_prior_blocks,
                              self.audio_spatial_blocks, self.audio_temporal_blocks, self.audio_st_prior_blocks,
                              self.va_cross_blocks)):
            # video spatio-temporal self-attention
            vx = auto_grad_checkpoint(v_s_blk, vx, y, t_mlp, y_lens, x_mask, t0_mlp, T, S)
            vx = auto_grad_checkpoint(v_t_blk, vx, y, t_mlp, y_lens, x_mask, t0_mlp, T, S)
            # audio spatio-temporal self-attention
            ax = auto_grad_checkpoint(a_s_blk, ax, ay, t_mlp, ay_lens, ax_mask, t0_mlp, R, M)
            ax = auto_grad_checkpoint(a_t_blk, ax, ay, t_mlp, ay_lens, ax_mask, t0_mlp, R, M)
            # video-prior spatio-temporal cross-attention
            vx = auto_grad_checkpoint(v_p_st_blk, vx, spatial_prior, temporal_prior, \
                                      t_st_mlp, None, x_mask, t0_st_mlp, T, S, return_attn_map=return_attn_map)
            if return_attn_map:
                vx, v_s_attn, v_t_attn = vx
                attn_maps[f'block{i}_video_spatial'] = _parse_attn_map(v_s_attn)
                attn_maps[f'block{i}_video_temporal'] = _parse_attn_map(v_t_attn)
            # audio-prior spatio-temporal cross-attention
            ax = auto_grad_checkpoint(a_p_st_blk, ax, spatial_prior, temporal_prior, \
                                      t_st_mlp, None, ax_mask, t0_st_mlp, R, M, return_attn_map=return_attn_map)
            if return_attn_map:
                ax, a_s_attn, a_t_attn = ax
                attn_maps[f'block{i}_audio_spatial'] = _parse_attn_map(a_s_attn)
                attn_maps[f'block{i}_audio_temporal'] = _parse_attn_map(a_t_attn)
            # video-audio spatio-temporal cross-attention  # TODO: mask
            vx, ax = auto_grad_checkpoint(va_st_blk, (vx, ax), (None, None))
            
        if self.enable_sequence_parallelism:
            vx = rearrange(vx, "B (T S) C -> B T S C", T=T, S=S)
            vx = gather_forward_split_backward(vx, get_sequence_parallel_group(), dim=2, grad_scale="up")
            S = S * dist.get_world_size(get_sequence_parallel_group())
            vx = rearrange(vx, "B T S C -> B (T S) C", T=T, S=S)
            ax = rearrange(ax, "B (T S) C -> B T S C", T=R, S=M)
            ax = gather_forward_split_backward(ax, get_sequence_parallel_group(), dim=2, grad_scale="up")
            M = M * dist.get_world_size(get_sequence_parallel_group())
            ax = rearrange(ax, "B T S C -> B (T S) C", T=R, S=M)

        # === final layer ===
        vx = self.final_layer(vx, t, x_mask, t0, T, S)
        vx = self.unpatchify(vx, T, H, W, Tx, Hx, Wx)
        ax = self.audio_final_layer(ax, t, ax_mask, t0, R, M)
        ax = self.unpatchify_audio(ax, R, M, Ta, Sa)

        # cast to float32 for better accuracy
        vx, ax = vx.to(torch.float32), ax.to(torch.float32)
        ret = {'video': vx, 'audio': ax}
        if return_attn_map:
            ret['attn_maps'] = attn_maps
        return ret

    def unpatchify_audio(self, x, N_t, N_s, R_t, R_s):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, S]
        """

        T_p, S_p = self.audio_patch_size
        x = rearrange(
            x,
            "B (N_t N_s) (T_p S_p C_out) -> B C_out (N_t T_p) (N_s S_p)",
            N_t=N_t,
            N_s=N_s,
            T_p=T_p,
            S_p=S_p,
            C_out=self.audio_out_channels,
        )
        # unpad
        x = x[:, :, :R_t, :R_s]
        return x


@MODELS.register_module("VASTDiT3-XL/2")
def VASTDiT3_XL_2(from_pretrained=None, **kwargs):
    if from_pretrained is not None and not os.path.isfile(from_pretrained):
        model = VASTDiT3.from_pretrained(from_pretrained, **kwargs)
    else:
        config = VASTDiT3Config(depth=28, hidden_size=1152, num_heads=16, **kwargs)
        model = VASTDiT3(config)
        if from_pretrained is not None:
            load_checkpoint(model, from_pretrained)
    return model


@MODELS.register_module("VASTDiT3-3B/2")
def VASTDiT3_3B_2(from_pretrained=None, **kwargs):
    if from_pretrained is not None and not os.path.isfile(from_pretrained):
        model = VASTDiT3.from_pretrained(from_pretrained, **kwargs)
    else:
        config = VASTDiT3Config(depth=28, hidden_size=1872, num_heads=26, **kwargs)
        model = VASTDiT3(config)
        if from_pretrained is not None:
            load_checkpoint(model, from_pretrained)
    return model
