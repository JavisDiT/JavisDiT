import os
from typing import Tuple, Optional, Literal
from copy import deepcopy
import gc
import math

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
    Attention, MultiHeadCrossAttention, BiMultiHeadAttention, SeqParallelAttention, SeqParallelMultiHeadCrossAttention,
    MMIdentity, MMZeros, Modulator, Additor, SizeEmbedder,
    CaptionEmbedder, PatchEmbed2D, PatchEmbed3D, PositionEmbedding, PositionEmbedding2D,
    T2IFinalLayer, approx_gelu, get_layernorm, t2i_modulate, ada_interpolate1d, smart_pad,
)
from javisdit.registry import MODELS
from javisdit.utils.ckpt_utils import load_checkpoint, load_ckpt_state_dict
from javisdit.utils.misc import get_logger, requires_grad
from javisdit.models.stdit.stdit3 import STDiT3Block, STDiT3, STDiT3Config

logger = get_logger()


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


class VASTDiT3BlockLayers(nn.Module):  # dummy class
    def __init__(
        self, hidden_size, num_heads, mlp_ratio, drop_path, rope, qk_norm,
        attn_cls, mha_cls, prior_attn_cls, enable_flash_attn, enable_layernorm_kernel
    ):  
        super().__init__()
        # spatio-temporal self-attention
        self.norm1 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.attn = attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=qk_norm,
            rope=rope,
            enable_flash_attn=enable_flash_attn,
        )
        # coarse-grained spatio-temporal cross-attention
        self.cross_attn = mha_cls(hidden_size, num_heads)
        # fine-grained spatio-temporal cross-attention
        self.prior_cross_attn = prior_attn_cls(hidden_size, num_heads)
        # mlp
        self.norm2 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
        )
        # modulate
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)
    
    def forward(self, x):  # dummy forward
        return x


class VASTDiT3Block(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        rope=None,
        qk_norm=False,
        temporal=False,
        prior_mode: Literal["cross_attn", "modulate", "addition"]='cross_attn',
        enable_flash_attn=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
    ):
        super().__init__()
        self.temporal = temporal
        self.enable_flash_attn = enable_flash_attn
        self.enable_layernorm_kernel = enable_layernorm_kernel
        self.enable_sequence_parallelism = enable_sequence_parallelism

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_path = drop_path
        self.rope = rope
        self.qk_norm = qk_norm

        if self.enable_sequence_parallelism and not temporal:
            attn_cls = SeqParallelAttention
            mha_cls = SeqParallelMultiHeadCrossAttention
        else:
            attn_cls = Attention
            mha_cls = MultiHeadCrossAttention

        self.prior_mode = prior_mode
        if prior_mode == 'cross_attn':
            prior_attn_cls = mha_cls
        elif prior_mode == 'modulate':
            prior_attn_cls = Modulator
        elif prior_mode == 'addition':
            prior_attn_cls = Additor
        
        self.video = VASTDiT3BlockLayers(
            hidden_size, num_heads, mlp_ratio, drop_path, rope, qk_norm,
            attn_cls, mha_cls, prior_attn_cls, enable_flash_attn, enable_layernorm_kernel
        )
        self.audio = VASTDiT3BlockLayers(
            hidden_size, num_heads, mlp_ratio, drop_path, rope, qk_norm,
            attn_cls, mha_cls, prior_attn_cls, enable_flash_attn, enable_layernorm_kernel
        )
        self.va_cross = CrossModalityBiAttentionBlock(
            hidden_size, hidden_size, hidden_size//2, num_heads//2,
            drop_path=drop_path, enable_layernorm_kernel=enable_layernorm_kernel,
            enable_flash_attn=enable_flash_attn
        )

        if self.temporal:
            self.video_onset_proj = Mlp(hidden_size, hidden_size)
            self.audio_onset_proj = Mlp(hidden_size, hidden_size)
    
    def apply_training_adjust(self, config: VASTDiT3Config):
        requires_grad(self, True)
        if config.freeze_video_branch:
            requires_grad(self.video, False)
        if config.freeze_audio_branch:
            requires_grad(self.audio, False)
        if config.only_train_audio or config.only_infer_audio:
            del self.video.attn, self.video.cross_attn, self.video.mlp
            self.video.attn = self.video.cross_attn = self.video.mlp = MMZeros()
        if not config.train_st_prior_attn:
            del self.video.prior_cross_attn, self.audio.prior_cross_attn
            self.video.prior_cross_attn = self.audio.prior_cross_attn = MMZeros()
        else:
            requires_grad(self.video.prior_cross_attn, True)
            requires_grad(self.audio.prior_cross_attn, True)
        if not config.train_va_cross_attn:
            del self.va_cross; self.va_cross = MMIdentity()

    def t_mask_select(self, x_mask, x, masked_x, T, S):
        # x: [B, (T, S), C]
        # mased_x: [B, (T, S), C]
        # x_mask: [B, T]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        masked_x = rearrange(masked_x, "B (T S) C -> B T S C", T=T, S=S)
        x = torch.where(x_mask[:, :, None, None], x, masked_x)
        x = rearrange(x, "B T S C -> B (T S) C")
        return x
    
    @staticmethod
    def get_modulate_params(branch: VASTDiT3BlockLayers, x_mask, t, t0, B):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            branch.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        prams = (
            shift_msa, scale_msa, gate_msa, 
            shift_mlp, scale_mlp, gate_mlp, 
        )

        if x_mask is not None:
            shift_msa_zero, scale_msa_zero, gate_msa_zero, shift_mlp_zero, scale_mlp_zero, gate_mlp_zero = (
                branch.scale_shift_table[None] + t0.reshape(B, 6, -1)
            ).chunk(6, dim=1)
            prams += (
                shift_msa_zero, scale_msa_zero, gate_msa_zero, 
                shift_mlp_zero, scale_mlp_zero, gate_mlp_zero, 
            )
        
        return prams
    
    def run_attn(self, branch: VASTDiT3BlockLayers, 
                 x, y, prior, mask, x_mask, modulate_params, T, S):
        # prepare
        shift_msa, scale_msa, gate_msa = modulate_params[0:3]
        if x_mask is not None:
            shift_msa_zero, scale_msa_zero, gate_msa_zero = modulate_params[6:9]
        
        # modulate (attention)
        x_m = t2i_modulate(branch.norm1(x), shift_msa, scale_msa)
        if x_mask is not None:
            x_m_zero = t2i_modulate(branch.norm1(x), shift_msa_zero, scale_msa_zero)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

        # spatio-temporal self-attention
        if self.temporal:
            x_m = rearrange(x_m, "B (T S) C -> (B S) T C", T=T, S=S)
            x_m = branch.attn(x_m)
            x_m = rearrange(x_m, "(B S) T C -> B (T S) C", T=T, S=S)
        else:
            x_m = rearrange(x_m, "B (T S) C -> (B T) S C", T=T, S=S)
            x_m = branch.attn(x_m)
            x_m = rearrange(x_m, "(B T) S C -> B (T S) C", T=T, S=S)

        # modulate (attention)
        x_m_s = gate_msa * x_m
        if x_mask is not None:
            x_m_s_zero = gate_msa_zero * x_m
            x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)

        # residual
        x = x + branch.drop_path(x_m_s)

        # coarse-grained spatio-temporal cross-attention
        x = x + branch.cross_attn(x, y, mask)

        # fine-grained spatio-temporal cross-attention
        # TODO: modulate or not 
        if self.temporal:
            x_m = rearrange(x, "B (T S) C -> (B S) T C", T=T, S=S)
            # prior: shape(B, T', C) -> shape(B*S, T', C) 
            prior = prior.unsqueeze(1).repeat(1, S, 1, 1).flatten(0,1)
            x_m = branch.prior_cross_attn(x_m, prior, None)
            x_m = rearrange(x_m, "(B S) T C -> B (T S) C", T=T, S=S)
        else:
            x_m = rearrange(x, "B (T S) C -> (B T) S C", T=T, S=S)
            # prior: shape(B, S', C) -> shape(B*T, S', C) 
            prior = prior.unsqueeze(1).repeat(1, T, 1, 1).flatten(0,1)
            x_m = branch.prior_cross_attn(x_m, prior, None)
            x_m = rearrange(x_m, "(B T) S C -> B (T S) C", T=T, S=S)

        x = x + x_m

        return x

    def run_mlp(self, branch: VASTDiT3BlockLayers, x, x_mask, modulate_params, T, S):
        # prepare
        shift_mlp, scale_mlp, gate_mlp = modulate_params[3:6]
        if x_mask is not None:
            shift_mlp_zero, scale_mlp_zero, gate_mlp_zero = modulate_params[9:12]

        # modulate (MLP)
        x_m = t2i_modulate(branch.norm2(x), shift_mlp, scale_mlp)
        if x_mask is not None:
            x_m_zero = t2i_modulate(branch.norm2(x), shift_mlp_zero, scale_mlp_zero)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

        # MLP
        x_m = branch.mlp(x_m)

        # modulate (MLP)
        x_m_s = gate_mlp * x_m
        if x_mask is not None:
            x_m_s_zero = gate_mlp_zero * x_m
            x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)

        # residual
        x = x + branch.drop_path(x_m_s)

        return x

    def add_onset_emb(self, x, onset_emb, T, S):
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        x = x + x * onset_emb.unsqueeze(2)  # (B,T,C) -> (B,T,1,C) 
        x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)
        return x

    def forward(
        self,
        x: Tuple[torch.Tensor, torch.Tensor],
        y: Tuple[torch.Tensor, torch.Tensor], 
        prior: Optional[torch.Tensor], 
        t: torch.Tensor,
        mask: Optional[Tuple[torch.Tensor, torch.Tensor]]=(None,None),  # text mask
        x_mask: Optional[Tuple[torch.Tensor, torch.Tensor]]=(None,None),  # video/audio temporal mask
        t0: Optional[torch.Tensor]=None,  # t with timestamp=0
        T: Optional[Tuple[int, int]] = None,  # number of video/audio frames
        S: Optional[Tuple[int, int]] = None,  # number of video/audio pixel patches
        onset: Optional[Tuple[torch.Tensor, torch.Tensor]]=(None, None),
    ):
        # prepare inputs
        vx, ax = x
        assert len(vx.shape) == len(ax.shape) == 3
        assert (B := vx.shape[0]) == ax.shape[0]
        assert (C := vx.shape[-1]) == ax.shape[-1]
        (Tv, Ta), (Sv, Sa) = T, S
        assert vx.shape[1] == Tv * Sv and ax.shape[1] == Ta * Sa
        vx_mask, ax_mask = x_mask
        (vy, ay), (v_mask, a_mask) = y, mask

        # add onset embedding
        vx_onset_emb, ax_onset_emb = onset
        if vx_onset_emb is not None and ax_onset_emb is not None:
            assert self.temporal
            vx_onset_emb = self.video_onset_proj(vx_onset_emb)
            ax_onset_emb = self.audio_onset_proj(ax_onset_emb)
            vx = self.add_onset_emb(vx, vx_onset_emb, Tv, Sv)
            ax = self.add_onset_emb(ax, ax_onset_emb, Ta, Sa)

        # prepare modulate parameters
        video_modulate_params = self.get_modulate_params(self.video, vx_mask, t, t0, B)
        audio_modulate_params = self.get_modulate_params(self.audio, ax_mask, t, t0, B)

        # single-modal spatio-temporal attention
        vx = self.run_attn(self.video, vx, vy, prior, v_mask, vx_mask, video_modulate_params, Tv, Sv)
        ax = self.run_attn(self.audio, ax, ay, prior, a_mask, ax_mask, audio_modulate_params, Ta, Sa)

        # cross-modal bidirectional attention
        vx, ax = self.va_cross((vx, ax), T=Tv, S=Sv, R=Ta, M=Sa)

        # mlp
        vx = self.run_mlp(self.video, vx, vx_mask, video_modulate_params, Tv, Sv)
        ax = self.run_mlp(self.audio, ax, ax_mask, audio_modulate_params, Ta, Sa)

        return vx, ax


class CrossModalityBiAttentionBlock(nn.Module):
    def __init__(self, m1_dim, m2_dim, hidden_size, num_heads, 
                 drop_path=0.0, enable_layernorm_kernel=False, enable_flash_attn=False,
                 init_values=1e-4, bica_mode:Literal['overall', 'temporal']='temporal'):
        super().__init__()
        self.m1_dim = m1_dim
        self.m2_dim = m2_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.attn_norm_m1 = get_layernorm(m1_dim, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.attn_norm_m2 = get_layernorm(m2_dim, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.cross_attn = BiMultiHeadAttention(
            m1_dim, m2_dim, hidden_size, num_heads, dropout=0.0,
            attn_implementation='flash_attn_2' if enable_flash_attn else 'sdpa'
        )

        self.gamma_m1 = nn.Parameter(init_values * torch.ones((m1_dim)), requires_grad=True)
        self.gamma_m2 = nn.Parameter(init_values * torch.ones((m2_dim)), requires_grad=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.bica_mode = bica_mode
    
    def forward(self, xs: Tuple[torch.Tensor], attention_masks: Optional[Tuple[torch.Tensor]]=(None,None),
                T: int=None, S: int=None, R: int=None, M: int=None):
        # x1: shape(B, T*S, C), x2: shape(B, R*M, C)
        x1, x2 = xs
        attention_mask_1, attention_mask_2 = attention_masks
        if attention_mask_1 is not None or attention_mask_2 is not None:
            raise NotImplementedError('attention mask is currently unsupported for video-audio cross attention')

        x_m1, x_m2 = self.attn_norm_m1(x1), self.attn_norm_m2(x2)

        if self.bica_mode == 'overall':
            dx_m1, dx_m2 = self.cross_attn(x_m1, x_m2, attention_mask_1, attention_mask_2)
        elif self.bica_mode == 'temporal':
            assert (B := x1.shape[0]) == x2.shape[0]
            assert (C := x1.shape[-1]) == x2.shape[-1]
            x_m1, x_m2 = x_m1.view(B, T, S, C), x_m2.view(B, R, M, C)
            # shape (B, R, M, C) -> (B, T, M*r, C)
            x_m2, x_m2_pad_mask = self.auto_temporal_slice(x_m2, pad_mask=None, window_num=T)
            # shape (B, T, S / M*r, C) -> (B*T, S / M*r, C)
            x_m1, x_m2 = x_m1.flatten(0, 1), x_m2.flatten(0, 1)
            attention_mask_2 = ~x_m2_pad_mask.flatten(0, 1)

            dx_m1, dx_m2 = self.cross_attn(x_m1, x_m2, None, attention_mask_2)

            dx_m1 = dx_m1.view(B, T*S, C)
            dx_m2 = dx_m2[attention_mask_2].view(B, R*M, C)
        else:
            raise NotImplementedError(self.bica_mode)

        x1 = x1 + self.drop_path(self.gamma_m1 * dx_m1)
        x2 = x2 + self.drop_path(self.gamma_m2 * dx_m2)

        return x1, x2

    def auto_temporal_slice(self, x: torch.Tensor, pad_mask: torch.Tensor, window_num: int):
        """
        Rearrange 1D padded tensor into a 2D tensor with zeros uniformly distributed.
        Thanks to DeepSeek-R1.
        
        Args:
            a (Tensor): Input tensor of shape (batch_size, length).
            mask (Tensor): Boolean mask tensor of shape (batch_size, length), True for valid elements.
            cols (int): Number of columns in the output 2D tensor.
        
        Returns:
            Tensor: Output tensor of shape (batch_size, rows, cols), where rows = length // cols.
        """
        # x: [B, T, S, C], pad_mask: [B, T, S]
        B, T, S, C = x.shape

        pad_len = math.ceil(T / window_num) * window_num - T
        if pad_len > 0:
            if pad_mask is None:
                pad_mask = torch.full(x.shape[:3], 0, dtype=torch.bool, device=x.device)
            x = smart_pad(x, pad_len, dim=1, mode='constant', value=0.)
            pad_mask = smart_pad(pad_mask, pad_len, dim=1, mode='constant', value=True)
        T += pad_len

        valid_mask = ~pad_mask[:, :, 0]  # [B, T] equal for (frequency component)
        window_size = T // window_num
        device = x.device
        
        # Flatten batch and sequence dimensions to handle all elements
        flat_mask = valid_mask.flatten()
        valid_elements = x.view(B*T, S*C)[flat_mask]  # (total_valid, Sa*C)
        
        # Compute indices for each valid element in the original tensor
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, T).reshape(-1)
        batch_indices = batch_indices[flat_mask]  # (total_valid,)
        
        # Calculate the number of valid elements per sample
        n_elements = valid_mask.sum(dim=1)  # (batch_size,)
        valid_n_elements = n_elements[batch_indices]
        
        # Calculate row and column indices for valid elements
        cum_counts = torch.cat([torch.zeros(1, device=device, dtype=torch.long), n_elements.cumsum(0)])
        local_indices = torch.arange(len(valid_elements), device=device) - cum_counts[batch_indices]
        
        # Compute row & col indices:
        rows_f = float(window_num)
        r = (local_indices.float() * rows_f / valid_n_elements.float()).floor().long()  # (total_valid,)
        k = (local_indices - r * valid_n_elements.float() / rows_f).floor().long()      # (total_valid,)
        
        # Ensure rows/columns do not exceed rows-1/cols-1
        valid_mask = (k < window_size) & (r < window_num)
        final_r = r[valid_mask]
        final_k = k[valid_mask]
        final_values = valid_elements[valid_mask]  # (total_valid, Sa*C)
        final_batch = batch_indices[valid_mask]
        
        # Create output tensor and scatter values
        output = torch.zeros((B, window_num, window_size, S*C), device=device, dtype=x.dtype)
        output_pad_mask = torch.ones((B, window_num, window_size, S), device=device, dtype=pad_mask.dtype)
        output[final_batch, final_r, final_k] = final_values
        output_pad_mask[final_batch, final_r, final_k] = 0

        output = output.view(B, window_num, window_size * S, C).contiguous()
        output_pad_mask = output_pad_mask.view(B, window_num, window_size * S).contiguous()
        
        return output, output_pad_mask


class VASTDiT3(STDiT3):
    config_class = VASTDiT3Config

    def __init__(self, config: VASTDiT3Config):
        super().__init__(config)
        # self.config: VASTDiT3Config
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
        self.spatial_blocks = nn.ModuleList(
            [
                VASTDiT3Block(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    drop_path=drop_path[i],
                    qk_norm=config.qk_norm,
                    prior_mode=config.st_prior_utilize,
                    enable_flash_attn=config.enable_flash_attn,
                    enable_layernorm_kernel=config.enable_layernorm_kernel,
                    enable_sequence_parallelism=config.enable_sequence_parallelism,
                )
                for i in range(config.depth)
            ]
        )

        # temporal blocks
        drop_path = [x.item() for x in torch.linspace(0, self.drop_path, config.depth)]
        self.temporal_blocks = nn.ModuleList(
            [
                VASTDiT3Block(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    drop_path=drop_path[i],
                    qk_norm=config.qk_norm,
                    prior_mode=config.st_prior_utilize,
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

        # final layer
        self.audio_final_layer = T2IFinalLayer(config.audio_hidden_size, 
                                               np.prod(self.audio_patch_size), 
                                               self.audio_out_channels)

        # initialize
        self.initialize_va_weights()
        # TODO: reuse null y_embedding
        self.audio_y_embedder.y_embedding.data = self.y_embedder.y_embedding.data

        # adjust modules
        self.apply_training_adjust(config)

    def initialize_va_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timporal blocks
        for block in self.temporal_blocks:
            # block: VASTDiT3Block
            for branch in [block.video, block.audio]:
                nn.init.constant_(branch.attn.proj.weight, 0)
                nn.init.constant_(branch.cross_attn.proj.weight, 0)
                nn.init.constant_(branch.prior_cross_attn.proj.weight, 0)
                nn.init.constant_(branch.mlp.fc2.weight, 0)
            nn.init.constant_(block.va_cross.cross_attn.out_m1_proj.weight, 0)
            nn.init.constant_(block.va_cross.cross_attn.out_m2_proj.weight, 0)
            nn.init.constant_(block.video_onset_proj.fc2.weight, 0)
            nn.init.constant_(block.audio_onset_proj.fc2.weight, 0)
                
        if init_dict := self.config.weight_init_from:
            model_dict = self.state_dict()
            state_dict = {}
            if init_audio_path := init_dict.get('audio', None):
                audio_dict = load_ckpt_state_dict(init_audio_path)
                for k, v in audio_dict.items():
                    if k in model_dict:
                        state_dict[k] = v
                    elif k.startswith('audio_'):
                        items = k.replace('audio_', '').split('.')
                        if items[2] != 'audio':
                            items.insert(2, 'audio')
                        k = '.'.join(items)
                        assert k in model_dict
                        state_dict[k] = v
                    else:
                        print('unrecognized audio', k)
                logger.info(f"{len(state_dict)}/{len(model_dict)} keys loaded from {init_audio_path}.")
            if init_video_path := init_dict.get('video', None):
                video_dict = load_ckpt_state_dict(init_video_path)
                for k, v in video_dict.items():
                    if k in state_dict:
                        assert v.shape == state_dict[k].shape
                        # if (mdelta := (v - state_dict[k]).abs().max()) >= 1e-4:
                        #     print(f'{k}: {mdelta}')
                        state_dict[k] = v
                    elif k in model_dict:
                        state_dict[k] = v
                    elif '_blocks.' in k:
                        items = k.split('.')
                        if items[2] != 'video':
                            items.insert(2, 'video')
                        k = '.'.join(items)
                        assert k in model_dict
                        state_dict[k] = v
                    else:
                        print('unrecognized video', k)
                logger.info(f"{len(state_dict)}/{len(model_dict)} keys loaded from {init_video_path}.")
            self.load_state_dict(state_dict, strict=False)

        # copy parameters from video to audio
        if self.config.only_train_audio:
            for va_blocks in [self.spatial_blocks, self.temporal_blocks]:
                for va_block in va_blocks:
                    # va_block: VASTDiT3Block
                    va_block.audio.load_state_dict(va_block.video.state_dict())
            self.audio_y_embedder.load_state_dict(deepcopy(self.y_embedder.state_dict()))
            logger.info('audio `spatial_blocks`, `temporal_blocks`, `y_embedder` are initialized from video branch.')

    def apply_training_adjust(self, config: VASTDiT3Config):
        if config.only_train_temporal or config.freeze_video_branch or config.freeze_audio_branch:
            # reused for two branches: t_block, fps_embedder t_embedder
            requires_grad(self, False)
            # if config.only_train_temporal:
            #     requires_grad(self.temporal_blocks, True)
            trainable_blocks = [self.y_embedder, self.audio_y_embedder, self.st_prior_embedder]
            if not config.freeze_video_branch:
                trainable_blocks.extend([self.x_embedder, self.final_layer])
            if not config.freeze_audio_branch:
                trainable_blocks.extend([self.ax_embedder, self.audio_final_layer])
            for block in trainable_blocks:
                requires_grad(block, True)
        
        if config.freeze_y_embedder:
            requires_grad(self.y_embedder, False)
            requires_grad(self.audio_y_embedder, False)

        if config.only_train_audio or config.only_infer_audio:
            requires_grad(self.y_embedder, False)
            requires_grad(self.final_layer, False)
        
        if config.only_train_temporal:
            main_blocks = self.temporal_blocks
        else:
            main_blocks = self.spatial_blocks + self.temporal_blocks
        for block in main_blocks:
            block.apply_training_adjust(config)

        if not config.train_st_prior_attn:
            del self.st_prior_embedder; self.st_prior_embedder = MMIdentity()
        
        gc.collect()

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
            st_prior = torch.cat((spatial_prior, temporal_prior), dim=1).to(dtype)
            st_prior = self.st_prior_embedder(st_prior.unsqueeze(1), self.training).squeeze(1)
            spatial_prior, temporal_prior = \
                st_prior.split([spatial_prior.shape[1], temporal_prior.shape[1]], dim=1)
        ax_mask = kwargs.pop('ax_mask', None)
        if ax_mask is not None:
            # TODO: stupid manual stride for ax_embedder
            ax_mask = ax_mask.view(B, -1, self.config.audio_patch_size[0]).any(dim=-1)
            # ax_mask = ax_mask[:, ::self.config.audio_patch_size[0]]  
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
        t_mlp = self.t_block(t)
        t0 = t0_mlp = None
        if x_mask is not None:
            assert ax_mask is not None
            t0_timestep = torch.zeros_like(timestep)
            t0 = self.t_embedder(t0_timestep, dtype=vx.dtype)
            t0 = t0 + fps
            t0_mlp = self.t_block(t0)

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
            vx_onset_emb = self.onset_embedder(vx_onset.squeeze(1), B).view(B, T, self.hidden_size)
            ax_onset_emb = self.onset_embedder(ax_onset.squeeze(1), B).view(B, R, self.hidden_size)
            # vx = vx + vx_onset_emb.unsqueeze(2)
            # ax = ax + ax_onset_emb.unsqueeze(2)
        else:
            vx_onset_emb, ax_onset_emb = None, None

        # shard over the sequence dim if sp is enabled
        if self.enable_sequence_parallelism:
            vx = split_forward_gather_backward(vx, get_sequence_parallel_group(), dim=2, grad_scale="down")
            S = S // dist.get_world_size(get_sequence_parallel_group())
            ax = split_forward_gather_backward(ax, get_sequence_parallel_group(), dim=2, grad_scale="down")
            M = M // dist.get_world_size(get_sequence_parallel_group())

        vx = rearrange(vx, "B T S C -> B (T S) C", T=T, S=S)
        ax = rearrange(ax, "B T S C -> B (T S) C", T=R, S=M)

        # === blocks ===
        for spatial_block, temporal_block in zip(self.spatial_blocks, self.temporal_blocks):
            vx, ax = auto_grad_checkpoint(
                spatial_block, 
                (vx, ax), (y, ay), spatial_prior, 
                t_mlp, (y_lens, ay_lens), (x_mask, ax_mask), t0_mlp, (T, R), (S, M)
            )
            vx, ax = auto_grad_checkpoint(
                temporal_block, 
                (vx, ax), (y, ay), temporal_prior, 
                t_mlp, (y_lens, ay_lens), (x_mask, ax_mask), t0_mlp, (T, R), (S, M),
                (vx_onset_emb, ax_onset_emb)
            )
            
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
        ret = {'video': vx.to(torch.float32), 'audio': ax.to(torch.float32)}

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


@MODELS.register_module("VASTDiT3_2-XL/2")
def VASTDiT3_2_XL_2(from_pretrained=None, **kwargs):
    if from_pretrained is not None and not os.path.isfile(from_pretrained):
        model = VASTDiT3.from_pretrained(from_pretrained, **kwargs)
    else:
        config = VASTDiT3Config(depth=28, hidden_size=1152, num_heads=16, **kwargs)
        model = VASTDiT3(config)
        if from_pretrained is not None:
            load_checkpoint(model, from_pretrained)
    return model


@MODELS.register_module("VASTDiT3_2-3B/2")
def VASTDiT3_2_3B_2(from_pretrained=None, **kwargs):
    if from_pretrained is not None and not os.path.isfile(from_pretrained):
        model = VASTDiT3.from_pretrained(from_pretrained, **kwargs)
    else:
        config = VASTDiT3Config(depth=28, hidden_size=1872, num_heads=26, **kwargs)
        model = VASTDiT3(config)
        if from_pretrained is not None:
            load_checkpoint(model, from_pretrained)
    return model
