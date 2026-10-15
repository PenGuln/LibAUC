# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.models.helpers import load_pretrained
from timm.models.vision_transformer import default_cfgs

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, n_classes=128, **kwargs):
        kwargs.pop('ada_ln_zero_init', None)  # remove this argument if it exists, since it's not used in this class
        allowed_kwargs = dict(kwargs)
        # Remove unsupported arguments
        unsupported_args = ['attention_lora_rank', 'attn_class', 'patch_embed_type', 'attention_layer', 'mlp_head', 'mlp_head_dim']
        for arg in unsupported_args:
            allowed_kwargs.pop(arg, None)
        super(VisionTransformer, self).__init__(num_classes=n_classes, **allowed_kwargs)

        self.global_pool = global_pool
        embed_dim = allowed_kwargs["embed_dim"]
        self.head = nn.Linear(embed_dim, n_classes) if n_classes > 0 else nn.Identity()
        if self.global_pool:
            norm_layer = allowed_kwargs['norm_layer']
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # remove the original norm

    def forward_features(self, x, attn_mask=None, **kwargs):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward(self, x, attn_mask=None, **kwargs):
        z = self.forward_features(x, attn_mask=attn_mask)
        return self.head(z)


"""
The settings of vit are from google's implementation:
https://github.com/google-research/vision_transformer/blob/main/vit_jax/configs/models.py

Currently, only the following models have ImageNet pretrained weights:
- ViT Base/16 (Pretrained on IN21K@14M and finetuned on IN1K@1M)
- ViT Large/16 (Pretrained on IN21K@14M and finetuned on IN1K@1M)
"""

def vit_tiny_patch16(pretrained=False, n_classes = 128, **kwargs):
    #assert timm.__version__ == "0.3.2"
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), n_classes=n_classes, **kwargs)
    if pretrained:
        raise ValueError("No pretrained model available for ViT Tiny")
    return model


def vit_small_patch16(pretrained=False, n_classes = 128, **kwargs):
    """
        Our small model is following the settings from google.
        It is different with timm's small model.
    """
    #assert timm.__version__ == "0.3.2"
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), n_classes=n_classes, **kwargs)
    if pretrained:
        raise ValueError("No pretrained model available for ViT Small")
    return model


def vit_base_patch16(pretrained=False, n_classes = 128,  **kwargs):
    #assert timm.__version__ == "0.3.2"
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), n_classes=n_classes, **kwargs)
    model.default_cfg = default_cfgs['vit_base_patch16_224']
    if pretrained:
        print("[INFO]: Use ImageNet pretrain vit-base/16 model.")
        from timm.models.vision_transformer import _conv_filter
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model


def vit_large_patch16(pretrained=False, n_classes = 128,  **kwargs):
    #assert timm.__version__ == "0.3.2"
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), n_classes=n_classes, **kwargs)
    model.default_cfg = default_cfgs['vit_large_patch16_224']
    if pretrained:
        print("[INFO]: Use ImageNet pretrain vit-large/16 model.")
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


def vit_huge_patch14(pretrained=False, n_classes = 128, **kwargs):
    #assert timm.__version__ == "0.3.2"
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), n_classes=n_classes, **kwargs)
    if pretrained:
        raise ValueError("No pretrained model available for ViT Huge")
    return model
