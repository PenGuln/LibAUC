# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

from functools import partial
import logging
import os
from typing import Callable, List, Any, Tuple, Dict
import warnings

import torch
from torch import nn, Tensor

from .attention import Attention, MemEffAttention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import fmha, scaled_index_add, index_select_cat

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Block)")
    else:
        warnings.warn("xFormers is disabled (Block)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False

    warnings.warn("xFormers is not available (Block)")


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        num_cond: int = 0,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            num_cond=num_cond,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor, cond_idx: torch.Tensor) -> Tensor:
        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x), cond_idx=cond_idx))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x


def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(
            x, brange, residual.to(dtype=x.dtype), scaling=scaling_vector, alpha=residual_scale_factor
        )
    return x_plus_residual


attn_bias_cache: Dict[Tuple, Any] = {}


def get_attn_bias_and_cat(x_list, branges=None):
    """
    this will perform the index select, cat the tensors, and provide the attn_bias from cache
    """
    batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensors


def drop_add_residual_stochastic_depth_list(
    x_list: List[Tensor],
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
) -> Tensor:
    # 1) generate random set of indices for dropping samples in the batch
    branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    # 2) get attention bias and index+concat the tensors
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)

    # 3) apply residual_func to get residual, and split the result
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

    outputs = []
    for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
    return outputs


class NestedTensorBlock(Block):
    def forward_nested(self, x_list: List[Tensor], cond_idx: List[Tensor]) -> List[Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """
        assert isinstance(self.attn, MemEffAttention)

        if self.training and self.sample_drop_ratio > 0.0:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.attn(self.norm1(x), attn_bias=attn_bias, cond_idx=cond_idx)

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.mlp(self.norm2(x))

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls1.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls2.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            return x_list
        else:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls2(self.mlp(self.norm2(x)))

            attn_bias, x = get_attn_bias_and_cat(x_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list, cond_idx):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list, cond_idx=cond_idx)
        elif isinstance(x_or_x_list, list):
            if not XFORMERS_AVAILABLE:
                raise AssertionError("xFormers is required for using nested tensors")
            return self.forward_nested(x_or_x_list, cond_idx=cond_idx)
        else:
            raise AssertionError

class ConditionedNestedTensorBlock(Block):
    def __init__(self, *args, cond_dim=128, **kwargs):
        # No affine params in norm1 and norm2 (per AdaLN-Zero)
        kwargs['norm_layer'] = partial(kwargs['norm_layer'], elementwise_affine=False)

        super().__init__(*args, **kwargs)

        # AdaLN modulation: generates 6 * dim parameters from conditioning vector
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * kwargs['dim'])
        )

        # self.norm1 = nn.LayerNorm(self.attn.qkv.in_features, elementwise_affine=False)
        # self.norm2 = nn.LayerNorm(self.attn.qkv.in_features, elementwise_affine=False)

    # modulation function
    def modulate(self, x, scale, shift):
        """Modulate normalized activations."""
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def compute_adaLN_params(self, cond):
        """Compute 6 conditioning vectors from cond input."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(cond).chunk(6, dim=-1)
        )
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    # new forward method
    def forward(self, x_or_x_list, cond: torch.Tensor, cond_idx: torch.Tensor):
        if isinstance(x_or_x_list, Tensor): 
            x = x_or_x_list
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.compute_adaLN_params(cond)

            def attn_residual_func(x: Tensor) -> Tensor:
                x_mod = self.modulate(self.norm1(x), scale_msa, shift_msa)
                return gate_msa.unsqueeze(1) * self.attn(x_mod, cond_idx=cond_idx)

            def ffn_residual_func(x: Tensor) -> Tensor:
                x_mod = self.modulate(self.norm2(x), scale_mlp, shift_mlp)
                return gate_mlp.unsqueeze(1) * self.mlp(x_mod)

            # Apply stochastic depth like in NeStBlock
            if self.training and self.sample_drop_ratio > 0.1:
                #print("Using high drop ratio path")
                x = drop_add_residual_stochastic_depth(
                    x, attn_residual_func, self.sample_drop_ratio
                )
                x = drop_add_residual_stochastic_depth(
                    x, ffn_residual_func, self.sample_drop_ratio
                )
            elif self.training and self.sample_drop_ratio > 0.0:
                #print("Using low drop ratio path")
                x = x + self.drop_path1(attn_residual_func(x))
                x = x + self.drop_path2(ffn_residual_func(x))
            else:
                #print("Using no drop path")
                x = x + attn_residual_func(x)
                x = x + ffn_residual_func(x)
            return x
            
        elif isinstance(x_or_x_list, list): #NOTE: in our case we will not use this path
            assert False, "Not tested"

            # Nested path (multi-view batching)
            assert isinstance(self.attn, MemEffAttention)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.compute_adaLN_params(cond)

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                x_mod = self.modulate(self.norm1(x), scale_msa, shift_msa)
                return gate_msa.unsqueeze(1) * self.attn(x_mod, attn_bias=attn_bias)

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                x_mod = self.modulate(self.norm2(x), scale_mlp, shift_mlp)
                return gate_mlp.unsqueeze(1) * self.mlp(x_mod)

            # (same stochastic depth logic)
            if self.training and self.sample_drop_ratio > 0.0:
                x_list = drop_add_residual_stochastic_depth_list(
                    x_list,
                    residual_func=attn_residual_func,
                    sample_drop_ratio=self.sample_drop_ratio,
                    scaling_vector=self.ls1.gamma if isinstance(self.ls1, LayerScale) else None,
                )
                x_list = drop_add_residual_stochastic_depth_list(
                    x_list,
                    residual_func=ffn_residual_func,
                    sample_drop_ratio=self.sample_drop_ratio,
                    scaling_vector=self.ls2.gamma if isinstance(self.ls1, LayerScale) else None,
                )
                return x_list
            else:
                attn_bias, x = get_attn_bias_and_cat(x_list)
                x = x + attn_residual_func(x, attn_bias=attn_bias)
                x = x + ffn_residual_func(x)
                return attn_bias.split(x)
        else:
            raise AssertionError


class FilmConditionedNestedTensorBlock(Block):
    def __init__(self, *args, cond_dim=128, **kwargs):
        # No affine params in norm1 and norm2 (per AdaLN-Zero)
        kwargs['norm_layer'] = partial(kwargs['norm_layer'], elementwise_affine=False)

        super().__init__(*args, **kwargs)

        # FiLM modulation: gamma/beta for attention and MLP branches.
        self.film_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 4 * self.attn.qkv.in_features)
        )

        # Keep affine-free norms to avoid competing learned affine transforms.
        # self.norm1 = nn.LayerNorm(self.attn.qkv.in_features, elementwise_affine=False)
        # self.norm2 = nn.LayerNorm(self.attn.qkv.in_features, elementwise_affine=False)

        # Identity initialization for FiLM at step 0.
        nn.init.zeros_(self.film_modulation[-1].weight)
        nn.init.zeros_(self.film_modulation[-1].bias)

    def _prepare_cond(self, cond: Tensor, batch_size: int, ref: Tensor) -> Tensor:
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        if cond.dim() != 2:
            raise ValueError(f"Expected cond with shape [B, C] or [C], got {tuple(cond.shape)}")
        if cond.shape[0] == 1 and batch_size > 1:
            cond = cond.expand(batch_size, -1)
        elif cond.shape[0] != batch_size:
            raise ValueError(f"cond batch={cond.shape[0]} must match x batch={batch_size} or be 1")
        return cond.to(device=ref.device, dtype=ref.dtype)

    def compute_film_params(self, cond: Tensor):
        gamma_msa, beta_msa, gamma_mlp, beta_mlp = self.film_modulation(cond).chunk(4, dim=-1)
        gamma_msa = 1 + torch.tanh(gamma_msa)
        gamma_mlp = 1 + torch.tanh(gamma_mlp)
        return gamma_msa, beta_msa, gamma_mlp, beta_mlp

    def film(self, y: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
        return y * gamma.unsqueeze(1) + beta.unsqueeze(1)

    def forward(self, x_or_x_list, cond: torch.Tensor, cond_idx: torch.Tensor):
        if isinstance(x_or_x_list, Tensor):
            x = x_or_x_list
            cond = self._prepare_cond(cond, x.shape[0], x)
            gamma_msa, beta_msa, gamma_mlp, beta_mlp = self.compute_film_params(cond)

            def attn_residual_func(x: Tensor) -> Tensor:
                y = self.ls1(self.attn(self.norm1(x), cond_idx=cond_idx))
                return self.film(y, gamma_msa, beta_msa)

            def ffn_residual_func(x: Tensor) -> Tensor:
                y = self.ls2(self.mlp(self.norm2(x)))
                return self.film(y, gamma_mlp, beta_mlp)

            # Apply stochastic depth like in NeStBlock
            if self.training and self.sample_drop_ratio > 0.1:
                x = drop_add_residual_stochastic_depth(
                    x, attn_residual_func, self.sample_drop_ratio
                )
                x = drop_add_residual_stochastic_depth(
                    x, ffn_residual_func, self.sample_drop_ratio
                )
            elif self.training and self.sample_drop_ratio > 0.0:
                x = x + self.drop_path1(attn_residual_func(x))
                x = x + self.drop_path2(ffn_residual_func(x))
            else:
                x = x + attn_residual_func(x)
                x = x + ffn_residual_func(x)
            return x
            
        elif isinstance(x_or_x_list, list):  # NOTE: in our case we will not use this path
            assert False, "Not tested"

            # Nested path (multi-view batching)
            assert isinstance(self.attn, MemEffAttention)
            x_list = x_or_x_list
            cond = self._prepare_cond(cond, x_list[0].shape[0], x_list[0])
            gamma_msa, beta_msa, gamma_mlp, beta_mlp = self.compute_film_params(cond)

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                y = self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))
                return self.film(y, gamma_msa, beta_msa)

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                y = self.ls2(self.mlp(self.norm2(x)))
                return self.film(y, gamma_mlp, beta_mlp)

            # (same stochastic depth logic)
            if self.training and self.sample_drop_ratio > 0.0:
                x_list = drop_add_residual_stochastic_depth_list(
                    x_list,
                    residual_func=attn_residual_func,
                    sample_drop_ratio=self.sample_drop_ratio,
                    scaling_vector=None,
                )
                x_list = drop_add_residual_stochastic_depth_list(
                    x_list,
                    residual_func=ffn_residual_func,
                    sample_drop_ratio=self.sample_drop_ratio,
                    scaling_vector=None,
                )
                return x_list
            else:
                attn_bias, x = get_attn_bias_and_cat(x_list)
                x = x + attn_residual_func(x, attn_bias=attn_bias)
                x = x + ffn_residual_func(x)
                return attn_bias.split(x)
        else:
            raise AssertionError
