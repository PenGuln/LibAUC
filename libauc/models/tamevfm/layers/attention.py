# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import math
import os
import warnings
import torch

from torch import Tensor
from torch import nn
import torch.nn.functional as F



XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
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


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, **kwargs) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SDPAAttention(Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._qkv = self.qkv  # keep the original qkv layer
        del self.qkv  # remove the original qkv layer to avoid confusion
        self.qkv = self.qkv_fn  # replace with a new qkv function that incorporates LoRA
    
    def qkv_fn(self, x: Tensor, cond_idx: Tensor) -> Tensor:
        return self._qkv(x)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        B, L, C = x.shape
        H = self.num_heads
        D = C // H

        qkv = self.qkv(x, **kwargs).reshape(B, L, 3, H, D).permute(2, 0, 3, 1, 4)  # [3, B, H, L, D]
        q, k, v = qkv.unbind(0)

        dropout_p = self.attn_drop.p if self.training else 0.0
        x = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=dropout_p,
            attn_mask=None,
            is_causal=False,
            scale=self.scale,
        )   # [B, H, L, D]
        x = x.transpose(1, 2).contiguous().view(B, L, C)  # [B, L, C]

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CondLoraAttention(SDPAAttention):
    def __init__(self, *args, num_cond=None, lora_rank=None, **kwargs):
        assert lora_rank is not None and lora_rank > 0, "Rank must be specified for LoRA attention"
        assert num_cond is not None and num_cond > 0, "Number of conditions must be specified for conditional attention"
        
        super().__init__(*args, **kwargs)
        self.lora_rank = int(lora_rank)

        ## get info from the original qkv linear layer
        dim = self._qkv.in_features
        qkv_bias = self._qkv.bias is not None

        ## initialize LoRA parameters
        self.A = nn.Parameter(torch.empty(num_cond, 3, dim, self.lora_rank))
        self.B = nn.Parameter(torch.empty(num_cond, 3, self.lora_rank, dim))
        
        ## initialize
        self._init_parameters()
    
    def _init_parameters(self):
        # initialize LoRA parameters
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.constant_(self.B, 0.0)

    def qkv_fn(self, x: Tensor, cond_idx: Tensor) -> Tensor:
        B, L, C = x.shape
        H = self.num_heads
        r = self.lora_rank

        ## Combine main weight and LoRA weights
        # [3C, C] + [3C, r] @ [r, C] -> [3C, C]
        qkv_base = self._qkv(x)  # [B, L, 3HD]

        # LoRA projection
        x = x.unsqueeze(1).expand(-1, 3, -1, -1)  # [B, 3, L, C]
        A = self.A[cond_idx]                          # [B, 3, C, r]
        Bm = self.B[cond_idx]                         # [B, 3, r, C]

        x = x.reshape(3*B, L, C)  # [3B, L, C]
        A = A.reshape(3*B, C, r)  # [3B, C, r]
        Bm = Bm.reshape(3*B, r, C)  # [3B, r, C]

        lora = torch.bmm(x, A)     # [B, 1, L, C] @ [B, 3, C, r] -> [B, 3, L, r]
        lora = torch.bmm(lora, Bm) # [B, 3, L, r] @ [B, 3, r, C] -> [B, 3, L, C]

        qkv = qkv_base + lora.view(B, 3, L, C).transpose(1,2).reshape(B, L, 3*C)  # [B, L, 3C]

        return qkv
