from typing import Optional, List, Dict, Literal, Union, Tuple
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from diffusers.utils import logging

from timm.models.layers import DropPath
from transformers import PretrainedConfig, PreTrainedModel

from javisdit.registry import MODELS
from javisdit.utils.misc import requires_grad
from javisdit.utils.ckpt_utils import load_checkpoint, load_ckpt_state_dict
from javisdit.models.layers.blocks import (
    approx_gelu, get_layernorm, t2i_modulate, CaptionEmbedder,
    PositionEmbedding, PositionEmbedding2D, PatchEmbed3D, 
    Attention, MultiHeadCrossAttention, BiMultiHeadAttention, Mlp
)
from javisdit.models.vae.utils import DiagonalGaussianDistribution


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class CrossModalityMergingBlock(nn.Module):
    def __init__(self, m1_dim, m2_dim, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0.0, 
                 temporal=False, qk_norm=False, enable_flash_attn=False, enable_layernorm_kernel=False):
        super().__init__()
        self.m1_dim = m1_dim
        self.m2_dim = m2_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.temporal = temporal

        self.attn_norm_m1 = get_layernorm(m1_dim, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.attn_norm_m2 = get_layernorm(m2_dim, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.cross_attn = BiMultiHeadAttention(m1_dim, m2_dim, hidden_size, num_heads, dropout=0.0)

        # self.attn_norm = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        # self.attn = Attention(
        #     hidden_size, num_heads=num_heads,
        #     qkv_bias=True, qk_norm=qk_norm, enable_flash_attn=enable_flash_attn
        # )

        self.mlp_norm = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), 
            act_layer=approx_gelu, drop=0.0
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor, attention_mask_1=None, attention_mask_2=None):
        # x1: shape(B, T, S, C), x2: shape(B, T, S, C)
        if self.temporal:
            assert x1.size(1) == x2.size(1)
        else:
            assert x1.size(2) == x2.size(2)
            x1, x2 = x1.transpose(1, 2).contiguous(), x2.transpose(1, 2).contiguous()
        B, N, _, C = x1.shape
        x1, x2 = x1.flatten(0, 1), x2.flatten(0, 1)
        assert x1.size(0) == x2.size(0) and x1.size(-1) == x2.size(-1)

        # x1: shape(B*N, L1, C), x2: shape(B*N, L2, C)

        # Cross Modality Attention
        x1_m, x2_m = self.attn_norm_m1(x1), self.attn_norm_m2(x2)
        x1_m, x2_m = self.cross_attn(x1_m, x2_m, attention_mask_1, attention_mask_2)
        x1, x2 = x1 + self.drop_path(x1_m), x2 + self.drop_path(x2_m)

        # Cross Modality Merging
        x = x1.mean(dim=1) + x2.mean(dim=1)
        x = x.view(B, N, C)
        # xm = self.attn_norm(x)
        # x = x + self.drop_path(self.attn(xm))

        # MLP
        x_m = self.mlp_norm(x)
        x = x + self.drop_path(self.mlp(x_m))

        return x  # shape(B, N, C)


class AttentionDiscriminator(nn.Module):
    def __init__(self, hidden_size, num_heads=4, qk_norm=False, enable_flash_attn=False):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.attn = Attention(dim=hidden_size, num_heads=num_heads, qkv_bias=True,
                              qk_norm=qk_norm, enable_flash_attn=enable_flash_attn)
        self.classifier = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor):
        # shape(B, N, C)
        x = torch.cat((self.cls_token.repeat(x.size(0), 1, 1), x), dim=1)
        z = self.attn(x)[:, 0, :]
        y = self.classifier(z).squeeze(-1)

        return y  # shape(B)


class STPriorExtractorConfig(PretrainedConfig):
    model_type = "STPriorExtractor"

    def __init__(
        self,
        ## for prior_encoder
        text_emb_dim: int = 1024,
        spatial_token_num: int = 32,
        temporal_token_num: int = 32,
        hidden_size: int = 512,
        out_dim: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        feedforward_scale: int = 4,
        dropout: float = 0.0,
        apply_sampling: bool = True,
        pred_onset: bool = False,
        ## for va_encoder
        encode_va: bool = False,
        video_in_channel: int = 4,
        video_patch_size: List[int] = (1, 2, 2),
        video_input_sq_size: int = 512,
        audio_in_channel: int = 8,
        audio_patch_size: List[int] = (2, 2),
        va_num_heads: int = 4,
        va_mlp_ratio: float = 4.0,
        va_drop_path: float = 0.0,
        qk_norm=False, enable_flash_attn=False, enable_layernorm_kernel=False,
        **kwargs
    ):
        ## for prior_encoder
        self.text_emb_dim = text_emb_dim
        self.spatial_token_num = spatial_token_num
        self.temporal_token_num = temporal_token_num
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.feedforward_scale = feedforward_scale
        self.dropout = dropout
        self.apply_sampling = apply_sampling
        self.pred_onset = pred_onset
        ## for va_encoder
        self.encode_va = encode_va
        self.video_in_channel = video_in_channel
        self.video_patch_size = video_patch_size
        self.video_input_sq_size = video_input_sq_size
        self.audio_in_channel = audio_in_channel
        self.audio_patch_size = audio_patch_size
        self.va_num_heads = va_num_heads
        self.va_mlp_ratio = va_mlp_ratio
        self.va_drop_path = va_drop_path
        self.qk_norm = qk_norm
        self.enable_flash_attn = enable_flash_attn
        self.enable_layernorm_kernel = enable_layernorm_kernel
        super().__init__(**kwargs)


class STPriorExtractor(PreTrainedModel):
    config_class = STPriorExtractorConfig

    """
    Spatio-Temporal Prior Extractor
    """
    def __init__(self, config: STPriorExtractorConfig):
        super().__init__(config)
        self.text_encoder = None
        self.config = config

        self.hidden_size = self.config.hidden_size
        self.spatial_token_num = self.config.spatial_token_num
        self.temporal_token_num = self.config.temporal_token_num

        self.spatial_query_emb = nn.Parameter(torch.randn(1, self.spatial_token_num, self.hidden_size))
        self.temporal_query_emb = nn.Parameter(torch.randn(1, self.temporal_token_num, self.hidden_size))

        self.in_proj = nn.Linear(self.config.text_emb_dim, self.hidden_size)
        self.prior_extractor = nn.Transformer(d_model=self.hidden_size, nhead=self.config.nhead,
                                              num_encoder_layers=self.config.num_encoder_layers, 
                                              num_decoder_layers=self.config.num_decoder_layers,
                                              dim_feedforward=self.hidden_size * self.config.feedforward_scale, 
                                              dropout=self.config.dropout, activation=approx_gelu(),
                                              batch_first=True, norm_first=True)
        
        if self.config.apply_sampling:
            self.out_proj = nn.Linear(self.hidden_size, self.config.out_dim*2)
        else:
            self.out_proj = nn.Linear(self.hidden_size, self.config.out_dim)
        
        if self.config.pred_onset:
            self.onset_predictor = nn.Linear(self.config.out_dim, 1)

        self.st_prior_embedder: CaptionEmbedder = None

        if self.config.encode_va:
            self.build_va_encoder()

    def encode_text(self, text: List[str]) -> torch.Tensor:
        raise NotImplementedError

    def free_text_encoder(self) -> None:
        if self.text_encoder is not None: 
            del self.text_encoder
    
    def encode(self, text: Union[str, List[str], torch.Tensor], calc_loss=False):
        if isinstance(text, str):
            text = [text]

        bs = len(text)

        if isinstance(text, torch.Tensor):
            text_hidden = text
        else:
            text_hidden = self.encode_text(text)  # shape(bs, 77, 1024)
        
        x = self.in_proj(text_hidden)
        st_query_embs = torch.cat((self.spatial_query_emb, self.temporal_query_emb), dim=1)
        st_prior = self.prior_extractor(x, st_query_embs.repeat(bs, 1, 1))
        st_prior = self.out_proj(st_prior)

        if self.config.apply_sampling:
            # [B, N, C]
            posterior = DiagonalGaussianDistribution(st_prior, dim=-1)
            st_prior = posterior.sample()
        else:
            posterior = None

        spatial_prior = st_prior[:, :self.spatial_token_num, :]
        temporal_prior = st_prior[:, self.spatial_token_num:, :]

        ret = {'spatial_prior': spatial_prior, 'temporal_prior': temporal_prior}

        if calc_loss:
            ret['posterior'] = posterior
        elif self.config.pred_onset:
            onset = self.onset_predictor(temporal_prior)  # shape(B,N,1)
            # onset = (onset >= 0.).float()  # hard
            onset = onset.sigmoid()        # soft
            ret['onset_prior'] = onset
        
        return ret

    def null(self, n):
        null_st_prior = self.st_prior_embedder.y_embedding[None].repeat(n, 1, 1)
        null_spatial_prior, null_temporal_prior = null_st_prior.split(
            [self.spatial_token_num, self.temporal_token_num], dim=1
        )
        return null_spatial_prior, null_temporal_prior

    def forward(self, text: Union[str, List[str], torch.Tensor], **kwargs):
        if kwargs.pop('mode', None) == 'calc_loss':
            ret = self.encode(text, calc_loss=True)
            return self.calc_prior_loss(
                ret['spatial_prior'], ret['temporal_prior'], ret['posterior'], **kwargs
            )
        else:
            return self.encode(text)

    def build_va_encoder(self):
        self.video_pos_embed = PositionEmbedding2D(self.config.out_dim)
        self.video_embedder = PatchEmbed3D(self.config.video_patch_size, self.config.video_in_channel, self.config.out_dim)

        self.audio_pos_embed = PositionEmbedding(self.config.out_dim)
        self.audio_embedder = nn.Conv2d(
            self.config.audio_in_channel, 
            self.config.out_dim, 
            kernel_size=self.config.audio_patch_size, 
            stride=self.config.audio_patch_size, 
            padding=0
        )

        self.va_spatial_encoder = CrossModalityMergingBlock(
            m1_dim=self.config.out_dim, 
            m2_dim=self.config.out_dim, 
            hidden_size=self.config.out_dim,
            num_heads=self.config.va_num_heads, 
            mlp_ratio=self.config.va_mlp_ratio, 
            drop_path=self.config.va_drop_path,
            temporal=False, 
            qk_norm=self.config.qk_norm, 
            enable_flash_attn=self.config.enable_flash_attn, 
            enable_layernorm_kernel=self.config.enable_layernorm_kernel
        )
        self.va_temporal_encoder = CrossModalityMergingBlock(
            m1_dim=self.config.out_dim, 
            m2_dim=self.config.out_dim, 
            hidden_size=self.config.out_dim,
            num_heads=self.config.va_num_heads, 
            mlp_ratio=self.config.va_mlp_ratio, 
            drop_path=self.config.va_drop_path,
            temporal=True, 
            qk_norm=self.config.qk_norm, 
            enable_flash_attn=self.config.enable_flash_attn,
            enable_layernorm_kernel=self.config.enable_layernorm_kernel
        )

        self.discriminator = AttentionDiscriminator(
            self.config.out_dim, 
            self.config.va_num_heads, 
            qk_norm=self.config.qk_norm, 
            enable_flash_attn=self.config.enable_flash_attn
        )
    
    def align_prior(self, x: torch.Tensor, mode: Literal['spatial', 'temporal']) -> torch.Tensor:
        B, T, S, C = x.shape

        if mode == 'spatial':
            x = rearrange(x, "B T S C -> (B T) C S", B=B, T=T, S=S, C=C)
            size = self.spatial_token_num
            x = F.interpolate(x, size=size, mode='linear', align_corners=False)
            x = rearrange(x, "(B T) C S -> B T S C", B=B, T=T, S=size, C=C)
        else:
            x = rearrange(x, "B T S C -> (B S) C T", B=B, T=T, S=S, C=C)
            size = self.temporal_token_num
            x = F.interpolate(x, size=size, mode='linear', align_corners=False)
            x = rearrange(x, "(B S) C T -> B T S C", B=B, T=size, S=S, C=C)
        
        return x
    
    def pre_encode_video(self, video: torch.Tensor, **kwargs) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # video: Tensor(B, C, T, H, W) -> Tensor(B, Tv, Sv, Cv)
        B, C, Tx, Hx, Wx = video.size()
        assert C == self.config.video_in_channel
        # assert all(s % p == 0 for s, p in zip([Tx, Hx, Wx], self.video_patch_size))
        # Tv, H, W = [s // p for s, p in zip([Tx, Hx, Wx], self.video_patch_size)]
        if Tx % self.config.video_patch_size[0] != 0:
            Tx += self.config.video_patch_size[0] - Tx % self.config.video_patch_size[0]
        if Hx % self.config.video_patch_size[1] != 0:
            Hx += self.config.video_patch_size[1] - Hx % self.config.video_patch_size[1]
        if Wx % self.config.video_patch_size[2] != 0:
            Wx += self.config.video_patch_size[2] - Wx % self.config.video_patch_size[2]
        Tv = Tx // self.config.video_patch_size[0]
        H = Hx // self.config.video_patch_size[1]
        W = Wx // self.config.video_patch_size[2]

        Sv = H * W
        frame_width, frame_height = kwargs.get('frame_width'), kwargs.get('frame_height')
        base_size = round(Sv**0.5)
        resolution_sq = (frame_height[0].item() * frame_width[0].item()) ** 0.5
        scale = resolution_sq / self.config.video_input_sq_size

        vx = self.video_embedder(video)  # [B, N, C]
        vx = rearrange(vx, "B (T S) C -> B T S C", T=Tv, S=Sv)
        pos_emb = self.video_pos_embed(vx, H, W, scale=scale, base_size=base_size)
        vx = vx + pos_emb

        vx_spatial = self.align_prior(vx, mode='spatial')
        vx_temporal = self.align_prior(vx, mode='temporal')

        return vx, vx_spatial, vx_temporal
    
    def pre_encode_audio(self, audio: torch.Tensor, **kwargs) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # audio: Tensor(B, 1, Ts, M) -> Tensor(B, Ta, Sa, Ca)
        B, _, Ts, M = audio.size()
        # assert Ts % self.config.audio_patch_size[0] == 0 and M % self.config.audio_patch_size[1] == 0  # patch size = 2
        # Ta, Sa = Ts // self.config.audio_patch_size[0], M // self.config.audio_patch_size[1]
        if Ts % self.config.audio_patch_size[0] != 0:
            Ts += self.config.audio_patch_size[0] - Ts % self.config.audio_patch_size[0]
        if M % self.config.audio_patch_size[1] != 0:
            M += self.config.audio_patch_size[1] - M % self.config.audio_patch_size[1]
        Ta = Ts // self.config.audio_patch_size[0]
        Sa = M // self.config.audio_patch_size[1]

        ax = self.audio_embedder(audio)
        ax = rearrange(ax, "B C T S -> B T S C", T=Ta, S=Sa)
        pos_emb = self.audio_pos_embed(ax, Sa)  # MelSpectrogram has fixed bins
        ax = ax + pos_emb

        ax_spatial = self.align_prior(ax, mode='spatial')
        ax_temporal = self.align_prior(ax, mode='temporal')

        return ax, ax_spatial, ax_temporal

    def calc_prior_loss(
        self, 
        spatial_prior: torch.Tensor,
        temporal_prior: torch.Tensor,
        posterior: Optional[DiagonalGaussianDistribution],
        video: torch.Tensor, 
        audio: torch.Tensor, 
        neg_videos: Dict[str, torch.Tensor],
        neg_audios: Dict[str, torch.Tensor],
        **kwargs
    ):
        """
        spatial_prior: Tensor(B, Ns, D),
        temporal_prior: Tensor(B, Nt, D),
        video: Tensor(B, Cv, T, H, W),
        audio: Tensor(B, Ca, Ts, M),
        neg_videos: Dict[str, Tensor(B, Nx, Cv, T, H, W)]
        neg_audios: Dict[str, Tensor(B, Nx, Ca, Ts, M)]
        """
        NEG_TYPES = ['spatial', 'temporal']
        assert all(neg_type in NEG_TYPES for neg_type in neg_videos.keys())
        assert all(neg_type in NEG_TYPES for neg_type in neg_audios.keys())

        B = video.shape[0]

        dtype, device = video.dtype, video.device
        empty_video = torch.zeros((B, 0, *video.shape[1:])).to(dtype=dtype, device=device)
        empty_audio = torch.zeros((B, 0, *audio.shape[1:])).to(dtype=dtype, device=device)

        spatial_neg_videos = neg_videos.get('spatial', empty_video)
        temporal_neg_videos = neg_videos.get('temporal', empty_video)
        spatial_neg_audios = neg_audios.get('spatial', empty_audio)
        temporal_neg_audios = neg_audios.get('temporal', empty_audio)

        num_spatial_neg_video, num_temporal_neg_video = spatial_neg_videos.shape[1], temporal_neg_videos.shape[1]
        num_spatial_neg_audio, num_temporal_neg_audio = spatial_neg_audios.shape[1], temporal_neg_audios.shape[1]
        num_neg_video = 1 + num_spatial_neg_video + num_temporal_neg_video
        num_neg_audio = 1 + num_spatial_neg_audio + num_temporal_neg_audio

        video_list = [video.unsqueeze(1), spatial_neg_videos, temporal_neg_videos]
        video_total = torch.cat(video_list, dim=1).flatten(0, 1)  # shape(B*Nv, C, T, H, W)
        audio_list = [audio.unsqueeze(1), spatial_neg_audios, temporal_neg_audios]
        audio_total = torch.cat(audio_list, dim=1).flatten(0, 1)  # shape(B*Na, 1, Ts, M)

        # shape(B, T, S, C)  --  T/S/C may be variant
        _, vx_spatial, vx_temporal = self.pre_encode_video(video_total, **kwargs)
        _, ax_spatial, ax_temporal = self.pre_encode_audio(audio_total, **kwargs)

        # find corresponding spatial negative and temporal negative
        vx_spatial = vx_spatial.view(B, num_neg_video, *vx_spatial.shape[1:])
        vx_temporal = vx_temporal.view(B, num_neg_video, *vx_temporal.shape[1:])
        pos_vx_s, neg_s_vx_s, neg_t_vx_s = vx_spatial.split([1, num_spatial_neg_video, num_temporal_neg_video], dim=1)
        pos_vx_t, neg_s_vx_t, neg_t_vx_t = vx_temporal.split([1, num_spatial_neg_video, num_temporal_neg_video], dim=1)
        ax_spatial = ax_spatial.view(B, num_neg_audio, *ax_spatial.shape[1:])
        ax_temporal = ax_temporal.view(B, num_neg_audio, *ax_temporal.shape[1:])
        pos_ax_s, neg_s_ax_s, neg_t_ax_s = ax_spatial.split([1, num_spatial_neg_audio, num_temporal_neg_audio], dim=1)
        pos_ax_t, neg_s_ax_t, neg_t_ax_t = ax_temporal.split([1, num_spatial_neg_audio, num_temporal_neg_audio], dim=1)

        vx_spatial_total = torch.cat(
            [pos_vx_s, pos_vx_s.repeat(1, num_spatial_neg_audio, 1, 1, 1), neg_s_vx_s], dim=1
        ).flatten(0, 1)
        ax_spatial_total = torch.cat(
            [pos_ax_s, neg_s_ax_s, pos_ax_s.repeat(1, num_spatial_neg_video, 1, 1, 1)], dim=1
        ).flatten(0, 1)
        assert vx_spatial_total.shape[0] == ax_spatial_total.shape[0]

        vx_temporal_total = torch.cat(
            [pos_vx_t, pos_vx_t.repeat(1, num_temporal_neg_audio, 1, 1, 1), neg_t_vx_t], dim=1
        ).flatten(0, 1)
        ax_temporal_total = torch.cat(
            [pos_ax_t, neg_t_ax_t, pos_ax_t.repeat(1, num_temporal_neg_video, 1, 1, 1)], dim=1
        ).flatten(0, 1)
        assert vx_temporal_total.shape[0] == ax_temporal_total.shape[0]

        # vx_spatial_pos, vx_spatial_neg = vx_spatial[:B, None], vx_spatial[B:].view(B, num_neg_video, **vx_spatial.shape[1:])
        # ax_spatial_pos, ax_spatial_neg = ax_spatial[:B, None], ax_spatial[B:].view(B, num_neg_audio, **ax_spatial.shape[1:])
        # vx_temporal_pos, vx_temporal_neg = vx_temporal[:B, None], vx_temporal[B:].view(B, num_neg_video, **vx_spatial.shape[1:])
        # ax_temporal_pos, ax_temporal_neg = ax_temporal[:B, None], ax_temporal[B:].view(B, num_neg_audio, **ax_spatial.shape[1:])

        # vx_spatial_total = torch.cat(
        #     [vx_spatial_pos, vx_spatial_pos.repeat(1, num_neg_audio, *video_dim_ones), vx_spatial_neg], dim=1
        # ).flatten(0, 1)
        # ax_spatial_total = torch.cat(
        #     [ax_spatial_pos, ax_spatial_neg, ax_spatial_pos.repeat(1, num_neg_video, *audio_dim_ones)], dim=1
        # ).flatten(0, 1)
        # assert vx_spatial_total.shape[0] == ax_spatial_total.shape[0]

        # vx_temporal_total = torch.cat(
        #     [vx_temporal_pos, vx_temporal_pos.repeat(1, num_neg_audio, *video_dim_ones), vx_temporal_neg], dim=1
        # ).flatten(0, 1)
        # ax_temporal_total = torch.cat(
        #     [ax_temporal_pos, ax_temporal_neg, ax_temporal_pos.repeat(1, num_neg_video, *audio_dim_ones)], dim=1
        # ).flatten(0, 1)
        # assert vx_temporal_total.shape[0] == ax_temporal_total.shape[0]
        
        spatial_embed = self.va_spatial_encoder(vx_spatial_total, ax_spatial_total)
        temporal_embed = self.va_temporal_encoder(vx_temporal_total, ax_temporal_total)

        spatial_embed = spatial_embed.view(B, 1+num_spatial_neg_video+num_spatial_neg_audio, *spatial_prior.shape[-2:])
        temporal_embed = temporal_embed.view(B, 1+num_temporal_neg_video+num_temporal_neg_audio, *temporal_prior.shape[-2:])

        losses, loss_dict = 0.0, {}
        losses += self.contrastive_loss(spatial_embed, spatial_prior, mode='spatial', loss_dict=loss_dict)
        losses += self.contrastive_loss(temporal_embed, temporal_prior, mode='temporal', loss_dict=loss_dict)
        if posterior is not None:
            kl_loss = 1e-6 * posterior.kl(dims=[1, 2]).mean()
            losses += kl_loss
            loss_dict['kl_loss'] = kl_loss.item()
        if self.config.pred_onset:
            onset_loss = self.onset_loss(temporal_prior, kwargs['onset'])
            losses += onset_loss
            loss_dict['onset_loss'] = onset_loss.item()
        
        return losses, loss_dict

    def onset_loss(self, temporal_prior, onset_label: torch.Tensor):
        if onset_label.shape[1] < self.temporal_token_num * 2:
            onset_label = self.align_prior(onset_label[:, :, None, None], mode='temporal').squeeze(-1)
        else:
            onset_label = F.adaptive_max_pool1d(onset_label[:, None, :], self.temporal_token_num).transpose(1,2)
        onset_pred = self.onset_predictor(temporal_prior)
        onset_loss = F.binary_cross_entropy_with_logits(onset_pred, onset_label)

        return onset_loss

    def contrastive_loss(self, embed: torch.Tensor, prior: torch.Tensor,
                         mode: Literal['spatial', 'temporal'], loss_dict: dict):
        cur_loss_dict: Dict[str, torch.Tensor] = {}

        loss_func = hinge_loss  # info_nce_loss, soft_margin_loss, margin_loss, hinge_loss

        # embed: shape(B, X, N, C); prior: shape(B, 1, N, C)
        prior = prior.unsqueeze(1)

        # 1. TokenWise
        embed_norm = F.normalize(embed, p=2, dim=-1)
        prior_norm = F.normalize(prior, p=2, dim=-1)
        similarity = (embed_norm * prior_norm).sum(-1)  # shape(B, X, N)
        token_sim_logit = similarity.transpose(1, 2).flatten(0,1) # shape(B*N, X)
        cur_loss_dict['token_loss'] = loss_func(token_sim_logit)
        
        # # 2. ElementWise
        # element_embed_norm = F.normalize(embed.flatten(-2,-1), p=2, dim=-1)
        # element_prior_norm = F.normalize(prior.flatten(-2,-1), p=2, dim=-1)
        # element_sim_logit = (element_embed_norm * element_prior_norm).sum(dim=-1)  # shape(B, X)
        # cur_loss_dict['element_loss'] = loss_func(element_sim_logit)

        # # 3. AvgPool
        # avg_embed_norm = F.normalize(embed.mean(dim=-2), p=2, dim=-1)
        # avg_prior_norm = F.normalize(prior.mean(dim=-2), p=2, dim=-1)
        # avg_sim_logit = (avg_embed_norm * avg_prior_norm).sum(dim=-1)  # shape(B, X)
        # cur_loss_dict['avg_loss'] = loss_func(avg_sim_logit)

        # 4. Discriminator
        B, X = embed.shape[:2]
        disc_logit = self.discriminator(
            torch.cat((prior.repeat(1, X, 1, 1), embed), dim=-2).flatten(0,1)
        )  # shape(B*X)
        disc_label = torch.zeros_like(disc_logit).view(B, X)
        disc_label[:, 0] = 1.
        cur_loss_dict['disc_loss'] = F.binary_cross_entropy_with_logits(disc_logit, disc_label.view(-1))

        # 5. L2-Norm
        cur_loss_dict['mse_loss'] = F.mse_loss(prior, embed[:, :1, :, :])

        # 6. VA-Margin
        pos_emb_norm, neg_emb_norm = embed_norm[:, :1, :, :], embed_norm[:, 1:, :, :]
        va_pair_logit = (pos_emb_norm * neg_emb_norm).sum(-1)  # shape(B, X-1, N)
        cur_loss_dict['va_loss'] = (va_pair_logit + 1.).relu().mean()  # all negative
       
        losses = sum(cur_loss_dict.values())
        loss_dict.update(
            {f'{mode}_{loss_type}': loss.item() for loss_type, loss in cur_loss_dict.items()}
        )
        loss_dict[f'{mode}_total_loss'] = losses.item()

        return losses
        
def info_nce_loss(logits, tau=0.5):
    labels = torch.zeros_like(logits[:, 0], dtype=torch.int64)
    return F.cross_entropy(logits / tau, labels)

def soft_margin_loss(logits):
    labels = torch.ones_like(logits)
    labels[:, 1:] = -1
    return F.soft_margin_loss(logits, labels)

def margin_loss(logits, margin=0.3):
    pos_logits, neg_logits = logits[:, :1], logits[:, 1:]
    return F.relu(neg_logits + margin - pos_logits).mean()

def hinge_loss(logits):
    pos_logits, neg_logits = logits[:, :1], logits[:, 1:]
    loss_pos = torch.mean(F.relu(1.0 - pos_logits))
    loss_neg = torch.mean(F.relu(1.0 + neg_logits))
    return 0.5 * (loss_pos + loss_neg)


@MODELS.register_module("BaseSTPrior")
def BaseSTPrior(from_pretrained=None, **kwargs):
    if from_pretrained is not None and not os.path.isfile(from_pretrained):
        model = STPriorExtractor.from_pretrained(from_pretrained, **kwargs)
    else:
        config = STPriorExtractorConfig(**kwargs)
        model = STPriorExtractor(config)
        if from_pretrained is not None:
            load_checkpoint(model, from_pretrained, strict=True)
    return model
