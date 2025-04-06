# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# PixArt: https://github.com/PixArt-alpha/PixArt-alpha
# Latte:  https://github.com/Vchitect/Latte
# DiT:    https://github.com/facebookresearch/DiT/tree/main
# GLIDE:  https://github.com/openai/glide-text2im
# MAE:    https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import functools
import math
from typing import Optional, Literal
import warnings

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
try:
    import xformers.ops
except:
    warnings.warn("install xformers to apply efficient attention computation")
from einops import rearrange
from timm.models.vision_transformer import Mlp
from timm.models.layers import DropPath

from javisdit.acceleration.communications import all_to_all, split_forward_gather_backward
from javisdit.acceleration.parallel_states import get_sequence_parallel_group

approx_gelu = lambda: nn.GELU(approximate="tanh")


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def get_layernorm(hidden_size: torch.Tensor, eps: float, affine: bool, use_kernel: bool):
    if use_kernel:
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(hidden_size, elementwise_affine=affine, eps=eps)
        except ImportError:
            raise RuntimeError("FusedLayerNorm not available. Please install apex.")
    else:
        return nn.LayerNorm(hidden_size, eps, elementwise_affine=affine)


def modulate(norm_func, x, shift, scale):
    # Suppose x is (B, N, D), shift is (B, D), scale is (B, D)
    dtype = x.dtype
    x = norm_func(x.to(torch.float32)).to(dtype)
    x = x * (scale.unsqueeze(1) + 1) + shift.unsqueeze(1)
    x = x.to(dtype)
    return x


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


def ada_interpolate1d(x, size, force_interpolate=False, mode='nearest', **kwargs):
    assert x.ndim == 3
    if x.shape[-1] == size:
        z = x
    elif x.shape[-1] < size or force_interpolate:
        z = F.interpolate(x, size=size, mode=mode, **kwargs)
    elif x.shape[-1] > size:
        z = F.adaptive_max_pool1d(x, size, **kwargs)
    return z


def smart_pad(x: torch.Tensor, pad_len, dim=0, mode="constant", value=0, 
              pos:Literal["right", "left", "both"]="right"):
    if dim < 0:
        dim += x.ndim
    assert dim < x.ndim, 'invalid padding dimension'
    pad_dim = [0, 0] * (x.ndim - dim - 1)
    if pos == "right":
        pad_dim += [0, pad_len]
    elif pos == "left":
        pad_dim += [pad_len, 0]
    else:
        pad_dim += [pad_len, pad_len]
    x = F.pad(x, pad_dim, mode=mode, value=value)
    return x


class Modulator(nn.Module):
    def __init__(self, hidden_size, num_heads=1, drop_path=0.0):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size*2)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(self, q: torch.Tensor, kv: torch.Tensor, mask=None):
        # q: shape(B,N1,C), kv: shape(B,N2,C)
        kv = kv.transpose(1,2).contiguous()
        kv = ada_interpolate1d(kv, size=q.shape[1], force_interpolate=True)
        kv = kv.transpose(1,2).contiguous()

        shift, scale = self.proj(kv).chunk(2, dim=-1)

        return q * scale + shift


class Additor(nn.Module):
    def __init__(self, hidden_size, num_heads=1, identity=False):
        super().__init__()
        if identity:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, q: torch.Tensor, kv: torch.Tensor, mask=None):
        # q: shape(B,N1,C), kv: shape(B,N2,C)
        kv = kv.transpose(1,2).contiguous()
        kv = ada_interpolate1d(kv, size=q.shape[1], force_interpolate=True)
        kv = kv.transpose(1,2).contiguous()

        return self.proj(kv)

# ===============================================
# General-purpose Layers
# ===============================================


class PatchEmbed3D(nn.Module):
    """Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        patch_size=(2, 4, 4),
        in_chans=3,
        embed_dim=96,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # (B C T H W)
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCTHW -> BNC
        return x


class PatchEmbed2D(nn.Module):
    """Audio to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2, 2).
        in_chans (int): Number of input audio channels. Default: 8.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        patch_size=(2, 2),
        in_chans=8,
        embed_dim=96,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1], 0, 0))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # (B C H W)
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = LlamaRMSNorm,
        enable_flash_attn: bool = False,
        rope=None,
        qk_norm_legacy: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flash_attn = enable_flash_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.qk_norm_legacy = qk_norm_legacy
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = False
        if rope is not None:
            self.rope = True
            self.rotary_emb = rope
        
        self.is_causal = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # flash attn is not memory efficient for small sequences, this is empirical
        enable_flash_attn = self.enable_flash_attn and (N > B)
        qkv = self.qkv(x)
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)

        qkv = qkv.view(qkv_shape).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.qk_norm_legacy:
            # WARNING: this may be a bug
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)
            q, k = self.q_norm(q), self.k_norm(k)
        else:
            q, k = self.q_norm(q), self.k_norm(k)
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)

        if enable_flash_attn:
            from flash_attn import flash_attn_func

            # (B, #heads, N, #dim) -> (B, N, #heads, #dim)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            x = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,
                causal=self.is_causal,
            )
        else:
            dtype = q.dtype
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # translate attn to float32
            attn = attn.to(torch.float32)
            if self.is_causal:
                causal_mask = torch.tril(torch.ones_like(attn), diagonal=0)
                causal_mask = torch.where(causal_mask.bool(), 0, float('-inf'))
                attn += causal_mask
            attn = attn.softmax(dim=-1)
            attn = attn.to(dtype)  # cast back attn to original dtype
            attn = self.attn_drop(attn)
            x = attn @ v

        x_output_shape = (B, N, C)
        if not enable_flash_attn:
            x = x.transpose(1, 2)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class KVCompressAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = LlamaRMSNorm,
        enable_flash_attn: bool = False,
        sampling="conv",
        sr_ratio=1,
        mem_eff_attention=False,
        attn_half=False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flash_attn = enable_flash_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.sr_ratio = sr_ratio
        self.sampling = sampling
        if sr_ratio > 1 and sampling == "conv":
            # Avg Conv Init.
            self.sr = nn.Conv2d(dim, dim, groups=dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr.weight.data.fill_(1 / sr_ratio**2)
            self.sr.bias.data.zero_()
            self.norm = nn.LayerNorm(dim)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.mem_eff_attention = mem_eff_attention
        self.attn_half = attn_half

    def downsample_2d(self, tensor, H, W, scale_factor, sampling=None):
        if sampling is None or scale_factor == 1:
            return tensor
        B, N, C = tensor.shape

        if sampling == "uniform_every":
            return tensor[:, ::scale_factor], int(N // scale_factor)

        tensor = tensor.reshape(B, H, W, C).permute(0, 3, 1, 2)
        new_H, new_W = int(H / scale_factor), int(W / scale_factor)
        new_N = new_H * new_W

        if sampling == "ave":
            tensor = F.interpolate(tensor, scale_factor=1 / scale_factor, mode="nearest").permute(0, 2, 3, 1)
        elif sampling == "uniform":
            tensor = tensor[:, :, ::scale_factor, ::scale_factor].permute(0, 2, 3, 1)
        elif sampling == "conv":
            tensor = self.sr(tensor).reshape(B, C, -1).permute(0, 2, 1)
            tensor = self.norm(tensor)
        else:
            raise ValueError

        return tensor.reshape(B, new_N, C).contiguous(), new_N

    def forward(self, x: torch.Tensor, mask=None, HW=None, block_id=None, **kwargs) -> torch.Tensor:
        B, N, C = x.shape
        new_N = N
        H, W = HW
        # flash attn is not memory efficient for small sequences, this is empirical
        enable_flash_attn = self.enable_flash_attn and (N > B)

        qkv = self.qkv(x).reshape(B, N, 3, C)
        q, k, v = qkv.unbind(2)
        dtype = q.dtype
        # KV compression
        if self.sr_ratio > 1:
            k, new_N = self.downsample_2d(k, H, W, self.sr_ratio, sampling=self.sampling)
            v, new_N = self.downsample_2d(v, H, W, self.sr_ratio, sampling=self.sampling)

        q = q.reshape(B, N, self.num_heads, C // self.num_heads).to(dtype)
        k = k.reshape(B, new_N, self.num_heads, C // self.num_heads).to(dtype)
        v = v.reshape(B, new_N, self.num_heads, C // self.num_heads).to(dtype)

        q, k = self.q_norm(q), self.k_norm(k)

        if enable_flash_attn:
            from flash_attn import flash_attn_func

            x = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,
            )

        elif self.mem_eff_attention:
            attn_bias = None
            if mask is not None:
                attn_bias = torch.zeros([B * self.num_heads, q.shape[1], k.shape[1]], dtype=q.dtype, device=q.device)
                attn_bias.masked_fill_(mask.squeeze(1).repeat(self.num_heads, 1, 1) == 0, float("-inf"))
            x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)
        else:
            # (B, N, #heads, #dim) -> (B, #heads, N, #dim)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            dtype = q.dtype
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # translate attn to float32
            if not self.attn_half:
                attn = attn.to(torch.float32)
            attn = attn.softmax(dim=-1)
            attn = attn.to(dtype)  # cast back attn to original dtype
            attn = self.attn_drop(attn)
            x = attn @ v

        x_output_shape = (B, N, C)
        if not enable_flash_attn:
            x = x.transpose(1, 2)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SeqParallelAttention(Attention):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = LlamaRMSNorm,
        enable_flash_attn: bool = False,
        rope=None,
    ) -> None:
        assert rope is None, "Rope is not supported in SeqParallelAttention"
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            enable_flash_attn=enable_flash_attn,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape  # for sequence parallel here, the N is a local sequence length
        qkv = self.qkv(x)
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.view(qkv_shape)

        sp_group = get_sequence_parallel_group()

        # apply all_to_all to gather sequence and split attention heads
        # [B, SUB_N, 3, NUM_HEAD, HEAD_DIM] -> [B, N, 3, NUM_HEAD_PER_DEVICE, HEAD_DIM]
        qkv = all_to_all(qkv, sp_group, scatter_dim=3, gather_dim=1)

        if self.enable_flash_attn:
            qkv_permute_shape = (
                2,
                0,
                1,
                3,
                4,
            )  # [3, B, N, NUM_HEAD_PER_DEVICE, HEAD_DIM]
        else:
            qkv_permute_shape = (
                2,
                0,
                3,
                1,
                4,
            )  # [3, B, NUM_HEAD_PER_DEVICE, N, HEAD_DIM]
        qkv = qkv.permute(qkv_permute_shape)

        # ERROR: Should qk_norm first
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.enable_flash_attn:
            from flash_attn import flash_attn_func

            x = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,
            )
        else:
            dtype = q.dtype
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # translate attn to float32
            attn = attn.to(torch.float32)
            attn = attn.softmax(dim=-1)
            attn = attn.to(dtype)  # cast back attn to original dtype
            attn = self.attn_drop(attn)
            x = attn @ v

        if not self.enable_flash_attn:
            x = x.transpose(1, 2)

        # apply all to all to gather back attention heads and split sequence
        # [B, N, NUM_HEAD_PER_DEVICE, HEAD_DIM]  -> [B, SUB_N, NUM_HEAD, HEAD_DIM]
        x = all_to_all(x, sp_group, scatter_dim=1, gather_dim=2)

        # reshape outputs back to [B, N, C]
        x_output_shape = (B, N, C)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0):
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond, mask=None, return_attn_map=False):
        # query/value: img tokens; key: condition; mask: if padding tokens
        B, N, C = x.shape

        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)

        attn_bias = None
        if mask is not None:
            attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
        if not return_attn_map:
            x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)
        else:
            attn_bias = attn_bias.materialize() if attn_bias is not None else None
            q = q.view(B, -1, self.num_heads, self.head_dim)
            k = k.view(B, -1, self.num_heads, self.head_dim)
            v = v.view(B, -1, self.num_heads, self.head_dim)
            x, attn = self.vanilla_attention(q, k, v, attn_bias=attn_bias)

        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attn_map:
            x = (x, attn)
        return x

    def vanilla_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_bias: torch.Tensor):
        ## q,k,v: shape(1, B*L, num_head, head_dim)
        scale = 1.0 / query.shape[-1] ** 0.5
        query = query * scale
        query = query.transpose(1, 2)  # shape (B, num_head, L, head_dim)
        key = key.transpose(1, 2)      # shape (B, num_head, L, head_dim)
        value = value.transpose(1, 2)  # shape (B, num_head, L, head_dim)
        attn = query @ key.transpose(-2, -1)    # shape (B, num_head, L, L)
        if attn_bias is not None:
            attn = attn + attn_bias
        attn_scores = attn.softmax(-1)
        attn_scores = self.attn_drop(attn_scores)    # shape (B, num_head, L, B*L)
        z = attn_scores @ value            # shape (B, num_head, L, head_dim)
        return z.transpose(1, 2).contiguous(), attn    # shape (B, L, num_head, head_dim)


class SeqParallelMultiHeadCrossAttention(MultiHeadCrossAttention):
    def __init__(
        self,
        d_model,
        num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__(
            d_model=d_model,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

    def forward(self, x, cond, mask=None):
        # query/value: img tokens; key: condition; mask: if padding tokens
        sp_group = get_sequence_parallel_group()
        sp_size = dist.get_world_size(sp_group)
        B, SUB_N, C = x.shape  # [B, TS/p, C]
        N = SUB_N * sp_size

        # shape:
        # q, k, v: [B, SUB_N, NUM_HEADS, HEAD_DIM]
        q = self.q_linear(x).view(B, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        kv = split_forward_gather_backward(kv, get_sequence_parallel_group(), dim=3, grad_scale="down")
        k, v = kv.unbind(2)

        # apply all_to_all to gather sequence and split attention heads
        q = all_to_all(q, sp_group, scatter_dim=2, gather_dim=1)

        q = q.view(1, -1, self.num_heads // sp_size, self.head_dim)
        k = k.view(1, -1, self.num_heads // sp_size, self.head_dim)
        v = v.view(1, -1, self.num_heads // sp_size, self.head_dim)

        # compute attention
        attn_bias = None
        if mask is not None:
            attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
        x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)

        # apply all to all to gather back attention heads and scatter sequence
        x = x.view(B, -1, self.num_heads // sp_size, self.head_dim)
        x = all_to_all(x, sp_group, scatter_dim=1, gather_dim=2)

        # apply output projection
        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, num_patch, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, num_patch * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final, x, shift, scale)
        x = self.linear(x)
        return x


class T2IFinalLayer(nn.Module):
    """
    The final layer of PixArt.
    """

    def __init__(self, hidden_size, num_patch, out_channels, d_t=None, d_s=None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, num_patch * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size**0.5)
        self.out_channels = out_channels
        self.d_t = d_t
        self.d_s = d_s

    def t_mask_select(self, x_mask, x, masked_x, T, S):
        # x: [B, (T, S), C]
        # mased_x: [B, (T, S), C]
        # x_mask: [B, T]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        masked_x = rearrange(masked_x, "B (T S) C -> B T S C", T=T, S=S)
        x = torch.where(x_mask[:, :, None, None], x, masked_x)
        x = rearrange(x, "B T S C -> B (T S) C")
        return x

    def forward(self, x, t, x_mask=None, t0=None, T=None, S=None):
        if T is None:
            T = self.d_t
        if S is None:
            S = self.d_s
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        if x_mask is not None:
            shift_zero, scale_zero = (self.scale_shift_table[None] + t0[:, None]).chunk(2, dim=1)
            x_zero = t2i_modulate(self.norm_final(x), shift_zero, scale_zero)
            x = self.t_mask_select(x_mask, x, x_zero, T, S)
        x = self.linear(x)
        return x


class BiMultiHeadAttention(nn.Module):
    def __init__(self, m1_dim, m2_dim, embed_dim, num_heads, dropout=0.0,
                 attn_implementation: Literal['eager', 'sdpa', 'flash_attn_2']='sdpa'):
        super(BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.m1_dim = m1_dim
        self.m2_dim = m2_dim

        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.m1_proj = nn.Linear(self.m1_dim, self.embed_dim)
        self.m2_proj = nn.Linear(self.m2_dim, self.embed_dim)
        self.values_m1_proj = nn.Linear(self.m1_dim, self.embed_dim)
        self.values_m2_proj = nn.Linear(self.m2_dim, self.embed_dim)

        self.out_m1_proj = nn.Linear(self.embed_dim, self.m1_dim)
        self.out_m2_proj = nn.Linear(self.embed_dim, self.m2_dim)

        self.stable_softmax_2d = True
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

        self._reset_parameters()
        self.attn_implementation = attn_implementation

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        for proj in [
            self.m1_proj, self.values_m1_proj, self.out_m1_proj, 
            self.m2_proj, self.values_m2_proj, self.out_m2_proj
        ]:
            nn.init.xavier_uniform_(proj.weight)
            proj.bias.data.fill_(0)

    def forward(self, x1, x2, attention_mask_1:torch.Tensor=None, attention_mask_2:torch.Tensor=None):
        """_summary_

        Args:
            x1 (_type_): bs, n_m1, dim
            x2 (_type_): bs, n_m2, dim
            attention_mask_1 (_type_, optional): _description_. bs, n_m1
            attention_mask_2 (_type_, optional): _description_. bs, n_m2

        Returns:
            _type_: _description_
        """
        attn_implementation = getattr(self, 'attn_implementation', 'eager')
        if attn_implementation == 'eager':
            return self.forward_eager(x1, x2, attention_mask_1, attention_mask_2)
        elif attn_implementation == 'sdpa':
            return self.forward_sdpa(x1, x2, attention_mask_1, attention_mask_2)
        elif attn_implementation == 'flash_attn_2':
            return self.forward_flash_attn_2(x1, x2, attention_mask_1, attention_mask_2)
        else:
            raise NotImplementedError(attn_implementation)
        
    def forward_eager(self, x1, x2, attention_mask_1:torch.Tensor=None, attention_mask_2:torch.Tensor=None):
        bsz, L1, _ = x1.size()
        device = x1.device

        query_states = self.m1_proj(x1) * self.scale                    # shape(B, L1, C)
        key_states = self._shape(self.m2_proj(x2), -1, bsz)             # shape(B, h, L2, d)
        value_m1_states = self._shape(self.values_m1_proj(x1), -1, bsz) # shape(B, h, L1, d)
        value_m2_states = self._shape(self.values_m2_proj(x2), -1, bsz) # shape(B, h, L2, d)

        # shape(B*h, L, d)
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, L1, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_m1_states = value_m1_states.view(*proj_shape)
        value_m2_states = value_m2_states.view(*proj_shape)

        L2 = key_states.size(1)  # L2
        # shape(B*h, L1, L2)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, L1, L2):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, L1, L2)}, "
                f"but is {attn_weights.size()}"
            )

        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()

        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(
                attn_weights, min=-50000
            )  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(
                attn_weights, max=50000
            )  # Do not increase 50000, data type half has quite limited range

        # shape(B*h, L2, L1)
        attn_weights_T = attn_weights.transpose(1, 2) 
        attn_weights_2 = attn_weights_T - torch.max(attn_weights_T, dim=-1, keepdim=True)[0]
        if self.clamp_min_for_underflow:
            attn_weights_2 = torch.clamp(
                attn_weights_2, min=-50000
            )  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights_2 = torch.clamp(
                attn_weights_2, max=50000
            )  # Do not increase 50000, data type half has quite limited range

        if attention_mask_1 is not None or attention_mask_2 is not None:
            if attention_mask_1 is None:
                attention_mask_1 = torch.ones((bsz, L1), dtype=attention_mask_2.dtype, device=device)
            if attention_mask_2 is None:
                attention_mask_2 = torch.ones((bsz, L2), dtype=attention_mask_1.dtype, device=device)
            # shape(L1, L2)
            mask_m2_to_m1 = attention_mask_1[:, :, None] | attention_mask_2[:, None, :]
            attn_weights.masked_fill_(torch.logical_not(mask_m2_to_m1), float("-inf"))
            # shape(L2, L1)
            mask_m1_to_m2 = mask_m2_to_m1.transpose(1, 2)
            attn_weights_2.masked_fill_(torch.logical_not(mask_m1_to_m2), float("-inf"))

        attn_probs_1 = attn_weights.softmax(dim=-1)
        attn_probs_2 = attn_weights_2.softmax(dim=-1)

        # shape(B*h, L1, L2)
        attn_probs_1 = F.dropout(attn_probs_1, p=self.dropout, training=self.training)
        # shape(B*h, L2, L1)
        attn_probs_2 = F.dropout(attn_probs_2, p=self.dropout, training=self.training)

        # shape(B*h, L1, L2) @ shape(B*h, L2, d) -> shape(B*h, L1, d)
        attn_output_1 = torch.bmm(attn_probs_1, value_m2_states)
        # shape(B*h, L2, L1) @ shape(B*h, L1, d) -> shape(B*h, L2, d)
        attn_output_2 = torch.bmm(attn_probs_2, value_m1_states)

        if attn_output_1.size() != (bsz * self.num_heads, L1, self.head_dim):
            raise ValueError(
                f"`attn_output_1` should be of size {(bsz, self.num_heads, L1, self.head_dim)}, "
                f"but is {attn_output_1.size()}"
            )

        if attn_output_2.size() != (bsz * self.num_heads, L2, self.head_dim):
            raise ValueError(
                f"`attn_output_2` should be of size {(bsz, self.num_heads, L2, self.head_dim)}, "
                f"but is {attn_output_2.size()}"
            )

        attn_output_1 = attn_output_1.view(bsz, self.num_heads, L1, self.head_dim)
        attn_output_1 = attn_output_1.transpose(1, 2)
        attn_output_1 = attn_output_1.reshape(bsz, L1, self.embed_dim)

        attn_output_2 = attn_output_2.view(bsz, self.num_heads, L2, self.head_dim)
        attn_output_2 = attn_output_2.transpose(1, 2)
        attn_output_2 = attn_output_2.reshape(bsz, L2, self.embed_dim)

        attn_output_1 = self.out_m1_proj(attn_output_1)
        attn_output_2 = self.out_m2_proj(attn_output_2)

        return attn_output_1, attn_output_2

    def forward_sdpa(self, x1, x2, attention_mask_1:torch.Tensor=None, attention_mask_2:torch.Tensor=None):
        bsz, L1, _ = x1.size()
        L2 = x2.size(1)
        device = x1.device

        query_states = self._shape(self.m1_proj(x1), -1, bsz)                # shape(B, h, L1, d)
        key_states = self._shape(self.m2_proj(x2), -1, bsz)                  # shape(B, h, L2, d)
        value_m1_states = self._shape(self.values_m1_proj(x1), -1, bsz)      # shape(B, h, L1, d)
        value_m2_states = self._shape(self.values_m2_proj(x2), -1, bsz)      # shape(B, h, L2, d)

        if attention_mask_1 is None and attention_mask_2 is None:
            mask_m1_to_m2, mask_m2_to_m1 = None, None
        else:
            if attention_mask_1 is None:
                attention_mask_1 = torch.ones((bsz, L1), dtype=attention_mask_2.dtype, device=device)
            if attention_mask_2 is None:
                attention_mask_2 = torch.ones((bsz, L2), dtype=attention_mask_1.dtype, device=device)
            # shape(L1, L2)
            mask_m2_to_m1 = attention_mask_1[:, None, :, None] | attention_mask_2[:, None, None, :]
            # shape(L2, L1)
            mask_m1_to_m2 = mask_m2_to_m1.transpose(-1, -2)
        
        attn_output_1 = F.scaled_dot_product_attention(
            query_states, key_states, value_m2_states,
            attn_mask=mask_m2_to_m1, dropout_p=self.dropout
        )
        attn_output_2 = F.scaled_dot_product_attention(
            key_states, query_states, value_m1_states,
            attn_mask=mask_m1_to_m2, dropout_p=self.dropout
        )

        attn_output_1 = attn_output_1.view(bsz, self.num_heads, L1, self.head_dim)
        attn_output_1 = attn_output_1.transpose(1, 2)
        attn_output_1 = attn_output_1.reshape(bsz, L1, self.embed_dim)

        attn_output_2 = attn_output_2.view(bsz, self.num_heads, L2, self.head_dim)
        attn_output_2 = attn_output_2.transpose(1, 2)
        attn_output_2 = attn_output_2.reshape(bsz, L2, self.embed_dim)

        attn_output_1 = self.out_m1_proj(attn_output_1)
        attn_output_2 = self.out_m2_proj(attn_output_2)

        return attn_output_1, attn_output_2

    def forward_flash_attn_2(self, x1, x2, attention_mask_1:torch.Tensor=None, attention_mask_2:torch.Tensor=None):
        bsz, L1, _ = x1.size()
        L2 = x2.size(1)

        if L1 <= bsz:  # copy from Attention block
            warnings.warn(f'Sequence length {L1} less than batch size {bsz}. Back to sdpa.')
            return self.forward_sdpa(x1, x2, attention_mask_1, attention_mask_2)

        if attention_mask_1 is not None and attention_mask_2 is not None:
            assert attention_mask_1.all() or attention_mask_2.all(), \
                'Currently does not support 2-directional mask attention'
        if attention_mask_1 is not None:
            x1 = x1 * attention_mask_1[..., None].type_as(x1)
        if attention_mask_2 is not None:
            x2 = x2 * attention_mask_2[..., None].type_as(x2)

        query_states = self._shape(self.m1_proj(x1), -1, bsz)                # shape(B, h, L1, d)
        key_states = self._shape(self.m2_proj(x2), -1, bsz)                  # shape(B, h, L2, d)
        value_m1_states = self._shape(self.values_m1_proj(x1), -1, bsz)      # shape(B, h, L1, d)
        value_m2_states = self._shape(self.values_m2_proj(x2), -1, bsz)      # shape(B, h, L2, d)

        from flash_attn import flash_attn_func

        # (B, #heads, N, #dim) -> (B, N, #heads, #dim)
        query_states = query_states.permute(0, 2, 1, 3)
        key_states = key_states.permute(0, 2, 1, 3)
        value_m1_states = value_m1_states.permute(0, 2, 1, 3)
        value_m2_states = value_m2_states.permute(0, 2, 1, 3)

        # (B, N, #heads, #dim) -> (B, #heads, N, #dim)
        attn_output_1 = flash_attn_func(
            query_states, key_states, value_m2_states,
            dropout_p=self.dropout if self.training else 0.0,
        ).transpose(1, 2)
        attn_output_2 = flash_attn_func(
            key_states, query_states, value_m1_states,
            dropout_p=self.dropout if self.training else 0.0,
        ).transpose(1, 2)

        attn_output_1 = attn_output_1.view(bsz, self.num_heads, L1, self.head_dim)
        attn_output_1 = attn_output_1.transpose(1, 2)
        attn_output_1 = attn_output_1.reshape(bsz, L1, self.embed_dim)

        attn_output_2 = attn_output_2.view(bsz, self.num_heads, L2, self.head_dim)
        attn_output_2 = attn_output_2.transpose(1, 2)
        attn_output_2 = attn_output_2.reshape(bsz, L2, self.embed_dim)

        attn_output_1 = self.out_m1_proj(attn_output_1)
        attn_output_2 = self.out_m2_proj(attn_output_2)

        return attn_output_1, attn_output_2


class MMIdentity(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, *args, **kwargs):
        return x

class MMZeros(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, *args, **kwargs):
        return torch.zeros_like(x)


# ===============================================
# Embedding Layers for Timesteps and Class Labels
# ===============================================


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        freqs = freqs.to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, dtype):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if t_freq.dtype != dtype:
            t_freq = t_freq.to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0]).cuda() < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class SizeEmbedder(TimestepEmbedder):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__(hidden_size=hidden_size, frequency_embedding_size=frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.outdim = hidden_size

    def forward(self, s, bs):
        if s.ndim == 1:
            s = s[:, None]
        assert s.ndim == 2
        if s.shape[0] != bs:
            s = s.repeat(bs // s.shape[0], 1)
            assert s.shape[0] == bs
        b, dims = s.shape[0], s.shape[1]
        s = rearrange(s, "b d -> (b d)")
        s_freq = self.timestep_embedding(s, self.frequency_embedding_size).to(self.dtype)
        s_emb = self.mlp(s_freq)
        s_emb = rearrange(s_emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim)
        return s_emb

    @property
    def dtype(self):
        return next(self.parameters()).dtype


class CaptionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(
        self,
        in_channels,
        hidden_size,
        uncond_prob,
        act_layer=nn.GELU(approximate="tanh"),
        token_num=120,
    ):
        super().__init__()
        self.y_proj = Mlp(
            in_features=in_channels,
            hidden_features=hidden_size,
            out_features=hidden_size,
            act_layer=act_layer,
            drop=0,
        )
        self.register_buffer(
            "y_embedding",
            torch.randn(token_num, in_channels) / in_channels**0.5,
        )
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0]).cuda() < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None, None], self.y_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        if train:
            assert caption.shape[2:] == self.y_embedding.shape, f'{caption.shape} {self.y_embedding.shape}'
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        caption = self.y_proj(caption)
        return caption


class PositionEmbedding2D(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        assert dim % 4 == 0, "dim must be divisible by 4"
        half_dim = dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, 2).float() / half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _get_sin_cos_emb(self, t: torch.Tensor):
        out = torch.einsum("i,d->id", t, self.inv_freq)
        emb_cos = torch.cos(out)
        emb_sin = torch.sin(out)
        return torch.cat((emb_sin, emb_cos), dim=-1)

    @functools.lru_cache(maxsize=512)
    def _get_cached_emb(
        self,
        device: torch.device,
        dtype: torch.dtype,
        h: int,
        w: int,
        scale: float = 1.0,
        base_size: Optional[int] = None,
    ):
        grid_h = torch.arange(h, device=device) / scale
        grid_w = torch.arange(w, device=device) / scale
        if base_size is not None:
            grid_h *= base_size / h
            grid_w *= base_size / w
        grid_h, grid_w = torch.meshgrid(
            grid_w,
            grid_h,
            indexing="ij",
        )  # here w goes first
        grid_h = grid_h.t().reshape(-1)
        grid_w = grid_w.t().reshape(-1)
        emb_h = self._get_sin_cos_emb(grid_h)
        emb_w = self._get_sin_cos_emb(grid_w)
        return torch.concat([emb_h, emb_w], dim=-1).unsqueeze(0).to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        h: int,
        w: int,
        scale: Optional[float] = 1.0,
        base_size: Optional[int] = None,
    ) -> torch.Tensor:
        return self._get_cached_emb(x.device, x.dtype, h, w, scale, base_size)


class PositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        assert dim % 2 == 0, "dim must be divisible by 4"
        half_dim = dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim).float() / half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _get_sin_cos_emb(self, t: torch.Tensor):
        out = torch.einsum("i,d->id", t, self.inv_freq)
        emb_cos = torch.cos(out)
        emb_sin = torch.sin(out)
        return torch.cat((emb_sin, emb_cos), dim=-1)
    
    @functools.lru_cache(maxsize=512)
    def _get_cached_emb(self, device: torch.device, dtype: torch.dtype, l: int):
        grid = torch.arange(l, device=device)
        emb = self._get_sin_cos_emb(grid)
        return emb.unsqueeze(0).to(dtype)

    def forward(self, x: torch.Tensor, l: int) -> torch.Tensor:
        return self._get_cached_emb(x.device, x.dtype, l)

# ===============================================
# Sine/Cosine Positional Embedding Functions
# ===============================================
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, scale=1.0, base_size=None):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / scale
    if base_size is not None:
        grid_h *= base_size / grid_size[0]
        grid_w *= base_size / grid_size[1]
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length, scale=1.0):
    pos = np.arange(0, length)[..., None] / scale
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


if __name__ == '__main__':
    from javisdit.utils.misc import Timer

    B, L1, L2, d = 1, 10, 12, 128
    dtype, device = torch.bfloat16, 'cuda:0'
    attn = BiMultiHeadAttention(
        128, 128, 128, 4, 0, attn_implementation='eager'
    ).eval().to(device=device, dtype=dtype)
    x1 = torch.randn((B, L1, d), dtype=dtype, device=device)
    x2 = torch.randn((B, L2, d), dtype=dtype, device=device)
    attn_mask_1 = torch.ones((B, L1), dtype=torch.bool, device=device)
    attn_mask_1[:, -2:] = 0
    attn_mask_2 = torch.ones((B, L2), dtype=torch.bool, device=device)
    # attn_mask_2[:, -3:] = 0

    with torch.no_grad():
        attn_res = {'w_mask': {}, 'wo_mask': {}}
        for attn_impl in ['eager', 'sdpa', 'flash_attn_2']:
            attn.attn_implementation = attn_impl
            with Timer(f'{attn_impl} w mask', log=True):
                attn_res['w_mask'][attn_impl] = attn(x1, x2, attn_mask_1, attn_mask_2)
            with Timer(f'{attn_impl} wo mask', log=True):
                attn_res['wo_mask'][attn_impl] = attn(x1, x2)

    import pdb; pdb.set_trace()
    pass
