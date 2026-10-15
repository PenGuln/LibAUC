
from functools import partial

import torch
import torch.nn as nn

from .layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block
from .vision_transformer import DinoVisionTransformer



class DinoVisionTransformerClassification(DinoVisionTransformer):

    def __init__(self, num_classes, global_pool=False, **kwargs):
        super(DinoVisionTransformerClassification, self).__init__(**kwargs)

        embed_dim = kwargs['embed_dim']
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        self.global_pool = global_pool
        if self.global_pool:
            # norm_layer = kwargs['norm_layer']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        x = self.prepare_tokens_with_masks(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, self.num_register_tokens+1:].mean(dim=1)
            out = self.fc_norm(x)
        else:
            x = self.norm(x)
            out = x[:, 0]

        return out

    # we overwrite the forward method for classification
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def dino_vit_tiny(num_classes=1000, global_pool=False, patch_size=16, num_register_tokens=0, **kwargs):
    """
    The settings of vit are not provided in the origion dinov2 implementation.
    We follow the settings of google's implementation:
    https://github.com/google-research/vision_transformer/blob/main/vit_jax/configs/models.py
    """
    model = DinoVisionTransformerClassification(
        num_classes=num_classes,
        global_pool=global_pool,
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def dino_vit_small(num_classes=1000, global_pool=False, patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformerClassification(
        num_classes=num_classes,
        global_pool=global_pool,
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def dino_vit_base(num_classes=1000, global_pool=False, patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformerClassification(
        num_classes=num_classes,
        global_pool=global_pool,
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def dino_vit_large(num_classes=1000, global_pool=False, patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformerClassification(
        num_classes=num_classes,
        global_pool=global_pool,
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model



def dino_vit_huge(num_classes=1000, global_pool=False, patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformerClassification(
        num_classes=num_classes,
        global_pool=global_pool,
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def dino_vit_giant2(num_classes=1000, global_pool=False, patch_size=16, num_register_tokens=0, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformerClassification(
        num_classes=num_classes,
        global_pool=global_pool,
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model
