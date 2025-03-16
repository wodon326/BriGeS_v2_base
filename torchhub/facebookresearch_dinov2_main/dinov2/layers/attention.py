# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging

from torch import Tensor
from torch import nn
import torch
import torch.nn.functional as F
from typing import Callable

def softmax_with_temperature(logits, tau=1.0):
    return F.softmax(logits / tau, dim=-1)

logger = logging.getLogger("dinov2")


try:
    from xformers.ops import memory_efficient_attention, unbind, fmha

    XFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False
from torchhub.facebookresearch_dinov2_main.dinov2.layers.mlp import GroupedLinear


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class GroupedAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_groups: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """
        dim: 임베딩 차원 (num_heads로 나누어떨어져야 함)
        num_heads: 멀티 헤드 수 (= 그룹 개수로 사용)
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # qkv: (dim -> dim*3) 변환을 그룹화
        self.qkv = GroupedLinear(
            in_features=dim,
            out_features=dim * 3,
            num_groups=num_groups,     # 보통 head 수와 동일하게 설정
            bias=qkv_bias
        )

        self.attn_drop = nn.Dropout(attn_drop)

        # proj: (dim -> dim) 변환을 그룹화
        self.proj = GroupedLinear(
            in_features=dim,
            out_features=dim,
            num_groups=num_groups,     # 동일하게 head 수
            bias=proj_bias
        )
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)  -> B: batch, N: sequence length, C: dim
        B, N, C = x.shape

        # (B, N, C) -> (B, N, 3*C)
        # reshape -> (B, N, 3, num_heads, head_dim)
        qkv = self.qkv(x)  # GroupedLinear
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)

        # (3, B, num_heads, N, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # q, k, v 분리 (모두 shape: (B, num_heads, N, head_dim))
        q, k, v = qkv[0], qkv[1], qkv[2]

        # scale 적용
        q = q * self.scale

        # 어텐션 스코어 (B, num_heads, N, N)
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # (B, num_heads, N, head_dim)
        out = torch.matmul(attn, v)

        # (B, N, num_heads, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)

        # 최종 투영
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class LoRA_Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rank : int = 64,
        act_layer: Callable[..., nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_lora_a = nn.Linear(dim, rank, bias=qkv_bias)
        self.qkv_act_layer = act_layer()
        self.qkv_lora_b = nn.Linear(rank, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_lora_a = nn.Linear(dim, rank, bias=proj_bias)
        self.proj_act_layer = act_layer()
        self.proj_lora_b = nn.Linear(rank, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv_lora_b(self.qkv_act_layer(self.qkv_lora_a(x))).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_lora_b(self.proj_act_layer(self.proj_lora_a(x)))
        x = self.proj_drop(x)
        return x
    
class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.query = nn.Linear(dim, dim, bias=qkv_bias)
        self.key_value = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        B, N, C = x.shape
        query = self.query(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        key_value = self.key_value(y).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = query[0] * self.scale, key_value[0], key_value[1]
        attn = q @ k.transpose(-2, -1)


        # attn_tau2 = softmax_with_temperature(attn, tau = 2)
        # attn_tau2 = self.attn_drop(attn_tau2)
        # x_tau2 = (attn_tau2 @ v).transpose(1, 2).reshape(B, N, C)
        # x_tau2 = self.proj(x_tau2)
        # x_tau2 = self.proj_drop(x_tau2)
        # torch.save(x_tau2, './softmax_tau2_x.pt')
        
        # softmax_tau_divide2 = softmax_with_temperature(attn, tau = 0.5)
        # softmax_tau_divide2 = self.attn_drop(softmax_tau_divide2)
        # x_tau_divide2 = (softmax_tau_divide2 @ v).transpose(1, 2).reshape(B, N, C)
        # x_tau_divide2 = self.proj(x_tau_divide2)
        # x_tau_divide2 = self.proj_drop(x_tau_divide2)
        # torch.save(x_tau_divide2, './softmax_tau_divide2_x.pt')
        
        # # softmax_tau_divide2 = softmax_with_temperature(attn, tau = 1/1.5)
        # # softmax_tau1 = softmax_with_temperature(attn, tau = 1)
        # # torch.save(softmax_tau1, './softmax_tau1.pt')
        # # torch.save(softmax_tau2, './softmax_tau2.pt')
        # # torch.save(softmax_tau_divide2, './softmax_tau_divide2.pt')


        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class Attention_tau(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1))

        
        # softmax_with_temperature(attn, tau = 2)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class CrossAttention_tau(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.query = nn.Linear(dim, dim, bias=qkv_bias)
        self.key_value = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        B, N, C = x.shape
        query = self.query(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        key_value = self.key_value(y).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = query[0] * self.scale, key_value[0], key_value[1]
        attn = (q @ k.transpose(-2, -1))


        # softmax_with_temperature(attn, tau = 1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            assert attn_bias is None, "xFormers is required for nested tensors usage"
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
