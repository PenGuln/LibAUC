# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

import logging
from typing import Callable, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
        self.proj = nn.Conv2d(
            in_chans, 
            embed_dim, 
            kernel_size=patch_HW, 
            stride=patch_HW
        )

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class ConditionedPatchEmbed(PatchEmbed):
    """
    Modified PatchEmbed block that uses different Conv2D layers:
    Same as original, but uses different Conv2D ('proj') layers 
    for each (modality, body_part) combination.
    """

    def __init__(
        self,
        num_combinations: int,
        # conditioning_map: Callable,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
            flatten_embedding=flatten_embedding,
        )
        self.num_combinations = num_combinations
        # self.conditioning_map = conditioning_map

        # ---- Create mapping for combinations ----
        self.proj = nn.ModuleList([
            nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )
            for _ in range(self.num_combinations)
        ])

    def forward(self, x: torch.Tensor, cond_idx: torch.Tensor):
            B = x.shape[0]

            ## compute the indices for each sample in the batch based on modality and body part
            # indices = self.conditioning_map(modality, body_part)  # (B,)

            ## precompute the output shape and allocate output tensor
            H_out, W_out = self.patches_resolution
            sample_out = self.proj[0](x[:1]) #to match the precision
            out = torch.empty(
                B,
                self.embed_dim,
                H_out,
                W_out,
                device=x.device,
                dtype=sample_out.dtype,
            )
            for idx in range(self.num_combinations):
                mask = cond_idx == idx
                if mask.any():
                    out[mask] = self.proj[idx](x[mask])
            x = out.flatten(2).transpose(1, 2)  # (B, N, D)
            x = self.norm(x)
            if not self.flatten_embedding:
                x = x.view(B, H_out, W_out, self.embed_dim)
            return x
            
    def flops(self) -> float:
        raise NotImplementedError("flops calculation for ConditionedPatchEmbed is not implemented yet")


class FasterConditionedPatchEmbed(ConditionedPatchEmbed):
    def __init__(
        self,
        num_combinations: int,
        # conditioning_map: Callable,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__(
            num_combinations=num_combinations,
            # conditioning_map=conditioning_map,
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
            flatten_embedding=flatten_embedding,
        )
        self.proj = None  # we won't use nn.Conv2d layers, but will directly learn the weights and biases

        ## Parameters for per-combination convolution
        self.shared_weights = nn.Parameter(torch.empty(self.embed_dim, self.in_chans, *self.patch_size))
        self.weights = nn.Parameter(torch.empty(self.num_combinations, embed_dim, in_chans, *self.patch_size))
        self.biases = nn.Parameter(torch.zeros(self.num_combinations, embed_dim))

        self._init_weights()

    def _init_weights(self):
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.weights.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w = self.shared_weights.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.biases, 0)

    def forward(self, x: torch.Tensor, cond_idx: torch.Tensor):
        B, C, H, W = x.shape

        ## Map to unique index for each combination
        # indices = self.conditioning_map(modality, body_part)  # (B,)

        ## Pick the right weights and biases for each sample in the batch based on modality and body part
        w_s = self.weights[cond_idx] + self.shared_weights  # (B, embed_dim, in_chans, patch_H, patch_W)
        b_s = self.biases[cond_idx]   # (B, embed_dim)

        ## Grouped convolution to apply different filters to each sample in the batch
        x_g = x.view(1, B * C, H, W)  # (1, B*in_chans, H, W)
        w_g = w_s.view(B * self.embed_dim, C, *self.patch_size)  # (B*embed_dim, in_chans, patch_H, patch_W)
        b_g = b_s.view(B * self.embed_dim)  # (B*embed_dim)
        out = F.conv2d(
            input=x_g,
            weight=w_g,  # (B, embed_dim, in_chans, patch_H, patch_W)
            bias=b_g,     # (B, embed_dim)
            stride=self.patch_size,
            groups=B,          # apply different filters to each sample in the batch
        )

        ## Reshape back to (B, embed_dim, H_out, W_out)
        out = out.view(B, self.embed_dim, *out.shape[-2:])  # (B, embed_dim, H_out, W_out)

        ## Flatten and normalize
        x = out.flatten(2).transpose(1, 2)  # (B, N, D)
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.view(B, out.shape[2], out.shape[3], self.embed_dim)
        
        return x
            
    def flops(self) -> float:
        raise NotImplementedError("flops calculation for ConditionedPatchEmbed is not implemented yet")


# class ModalityConditionedPatchEmbed(nn.Module):
#     """
#     Modified PatchEmbed block that uses different Conv2D layers:
#     Same as original, but uses different Conv2D ('proj') layers 
#     for each modality.
#     """

#     def __init__(
#         self,
#         modalities: list,
#         img_size: Union[int, Tuple[int, int]] = 224,
#         patch_size: Union[int, Tuple[int, int]] = 16,
#         in_chans: int = 3,
#         embed_dim: int = 768,
#         norm_layer: Optional[Callable] = None,
#         flatten_embedding: bool = True,
#     ) -> None:
#         super().__init__()

#         image_HW = make_2tuple(img_size)
#         patch_HW = make_2tuple(patch_size)
#         patch_grid_size = (
#             image_HW[0] // patch_HW[0],
#             image_HW[1] // patch_HW[1],
#         )

#         self.img_size = image_HW
#         self.patch_size = patch_HW
#         self.patches_resolution = patch_grid_size
#         self.num_patches = patch_grid_size[0] * patch_grid_size[1]
#         self.in_chans = in_chans
#         self.embed_dim = embed_dim
#         self.flatten_embedding = flatten_embedding


#         # book-keeping
#         self.modalities = modalities
#         self.num_modalities = len(modalities)
#         self.num_combinations = self.num_modalities

#         self.proj = nn.ModuleList([
#             nn.Conv2d(
#                 in_chans,
#                 embed_dim,
#                 kernel_size=patch_HW,
#                 stride=patch_HW,
#             )
#             for _ in range(self.num_combinations)
#         ])

#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

#     def forward(self, x: torch.Tensor, modality: torch.Tensor):
#             B = x.shape[0]
#             H_out, W_out = self.patches_resolution
            
#             sample_out = self.proj[0](x[:1]) #to match the precision
#             out = torch.empty(
#                 B,
#                 self.embed_dim,
#                 H_out,
#                 W_out,
#                 device=x.device,
#                 dtype=sample_out.dtype,
#             )
#             for idx in range(self.num_combinations):
#                 mask = modality == idx
#                 if mask.any():
#                     out[mask] = self.proj[idx](x[mask])
#             x = out.flatten(2).transpose(1, 2)  # (B, N, D)
#             x = self.norm(x)
#             if not self.flatten_embedding:
#                 x = x.view(B, H_out, W_out, self.embed_dim)
#             return x
            
#     def flops(self) -> float:
#         Ho, Wo = self.patches_resolution
#         flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
#         if self.norm is not None:
#             flops += Ho * Wo * self.embed_dim
#         return flops
