
from functools import partial

import torch
import torch.nn as nn

from .layers import Mlp, ConditionedPatchEmbed, SwiGLUFFNFused, MemEffAttention, ConditionedNestedTensorBlock as Block
from .layers import SharedConditionedLinear, SharedConditionedMlpHead
from .vision_transformer import ConditionedViT
from .layers import attention as attn_layers


class ConditionedViTClassification(ConditionedViT):
    def __init__(self, modalities, body_parts, num_classes, global_pool=False, mlp_head=False, mlp_head_dim=2048, conditioned_head=False, **kwargs):
        assert 'pretrained' not in kwargs, "ConditionedViTClassification does not support loading pretrained weights yet"
        super(ConditionedViTClassification, self).__init__(modalities=modalities, body_parts=body_parts, **kwargs)

        ## Ensure head is identity before we overwrite it, to avoid accidentally overwriting
        assert isinstance(self.head, nn.Identity), "ConditionedViTClassification expects the head to be an identity layer, but got {}".format(type(self.head))
        ## Initialize head to contrastive projection
        embed_dim = kwargs['embed_dim']
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        ## Whether the head takes cond_idx (shared + modality-specific weights) or not
        self.head_is_conditioned = conditioned_head and num_classes > 0

        if num_classes <= 0:
            self.head = nn.Identity()
        elif conditioned_head and mlp_head:
            self.head = SharedConditionedMlpHead(
                num_combinations=self.num_conditioning_combinations,
                in_features=embed_dim,
                hidden_features=mlp_head_dim,
                out_features=num_classes,
            )
        elif conditioned_head:
            self.head = SharedConditionedLinear(
                num_combinations=self.num_conditioning_combinations,
                in_features=embed_dim,
                out_features=num_classes,
            )
        elif mlp_head:
            self.head = nn.Sequential(
                nn.Linear(embed_dim, mlp_head_dim),
                nn.LayerNorm(mlp_head_dim),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_head_dim, num_classes),
            )
        else:
            self.head = nn.Linear(embed_dim, num_classes)

        ## Whether to use global pool instead of CLS token
        self.global_pool = global_pool
        if self.global_pool:
            raise NotImplementedError("Global pool not implemented yet for ConditionedViTClassification")
            # norm_layer = kwargs['norm_layer']
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # remove the original norm

    def forward(self, x, modality=None, body_part=None, masks=None, return_intermediate=False, **kwargs):
        ret = self.forward_features(x, modality=modality, body_part=body_part, masks=masks)

        if return_intermediate:
            return ret

        cls_token = ret["x_norm_clstoken"]
        if self.head_is_conditioned:
            return self.head(cls_token, ret["cond_idx"])
        return self.head(cls_token)


# below are the tamevfm models
def tamevfm_dino_vit_tiny(num_classes=1000, global_pool=False, patch_size=16, num_register_tokens=0, modalities=None, body_parts=None, **kwargs):
    """
    The settings of vit are not provided in the origion dinov2 implementation.
    We follow the settings of google's implementation:
    https://github.com/google-research/vision_transformer/blob/main/vit_jax/configs/models.py
    """
    if 'attn_class' in kwargs:
        attn_class_name = kwargs.pop('attn_class')
        attn_class = getattr(attn_layers, attn_class_name)
        attn_class = partial(attn_class, lora_rank=kwargs.pop('attention_lora_rank'))
    else:
        attn_class = MemEffAttention
    
    model = ConditionedViTClassification(
        modalities = modalities,
        body_parts = body_parts,
        num_classes=num_classes,
        global_pool=global_pool,
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=attn_class),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def tamevfm_dino_vit_small(num_classes=1000, global_pool=False, patch_size=16, num_register_tokens=0, modalities=None, body_parts=None, **kwargs):
    if 'attn_class' in kwargs:
        attn_class_name = kwargs.pop('attn_class')
        attn_class = getattr(attn_layers, attn_class_name)
        attn_class = partial(attn_class, lora_rank=kwargs.pop('attention_lora_rank'))
    else:
        attn_class = MemEffAttention
    
    model = ConditionedViTClassification(
        modalities = modalities,
        body_parts = body_parts,
        num_classes=num_classes,
        global_pool=global_pool,
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=attn_class),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def tamevfm_dino_vit_base(num_classes=1000, global_pool=False, patch_size=16, num_register_tokens=0, modalities=None, body_parts=None, **kwargs):
    if 'attn_class' in kwargs:
        attn_class_name = kwargs.pop('attn_class')
        attn_class = getattr(attn_layers, attn_class_name)
        attn_class = partial(attn_class, lora_rank=kwargs.pop('attention_lora_rank'))
    else:
        attn_class = MemEffAttention
    kwargs.pop('attention_lora_rank', None)
    
    model = ConditionedViTClassification(
        modalities = modalities,
        body_parts = body_parts,
        num_classes=num_classes,
        global_pool=global_pool,
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        # todo! flash attention faster?
        block_fn=partial(Block, attn_class=attn_class),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def tamevfm_dino_vit_large(num_classes=1000, global_pool=False, patch_size=16, num_register_tokens=0, modalities=None, body_parts=None, **kwargs):
    if 'attn_class' in kwargs:
        attn_class_name = kwargs.pop('attn_class')
        attn_class = getattr(attn_layers, attn_class_name)
        attn_class = partial(attn_class, lora_rank=kwargs.pop('attention_lora_rank'))
    else:
        attn_class = MemEffAttention
    kwargs.pop('attention_lora_rank', None)
    
    model = ConditionedViTClassification(
        modalities = modalities,
        body_parts = body_parts,
        num_classes=num_classes,
        global_pool=global_pool,
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=attn_class),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model



def tamevfm_dino_vit_huge(num_classes=1000, global_pool=False, patch_size=16, num_register_tokens=0, modalities=None, body_parts=None, **kwargs):
    if 'attn_class' in kwargs:
        attn_class_name = kwargs.pop('attn_class')
        attn_class = getattr(attn_layers, attn_class_name)
        attn_class = partial(attn_class, lora_rank=kwargs.pop('attention_lora_rank'))
    else:
        attn_class = MemEffAttention
    kwargs.pop('attention_lora_rank', None)
    
    model = ConditionedViTClassification(
        modalities = modalities,
        body_parts = body_parts,
        num_classes=num_classes,
        global_pool=global_pool,
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=attn_class),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def tamevfm_dino_vit_giant2(num_classes=1000, global_pool=False, patch_size=16, num_register_tokens=0, modalities=None, body_parts=None, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    if 'attn_class' in kwargs:
        attn_class_name = kwargs.pop('attn_class')
        attn_class = getattr(attn_layers, attn_class_name)
        attn_class = partial(attn_class, lora_rank=kwargs.pop('attention_lora_rank'))
    else:
        attn_class = MemEffAttention
    kwargs.pop('attention_lora_rank', None)

    model = ConditionedViTClassification(
        modalities = modalities,
        body_parts = body_parts,
        num_classes=num_classes,
        global_pool=global_pool,
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=attn_class),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model



############ NEW SET ###############

#below are the modality conditioned vfm models
def modality_vfm_dino_vit_tiny(**kwargs):
    """
    The settings of vit are not provided in the origion dinov2 implementation.
    We follow the settings of google's implementation:
    https://github.com/google-research/vision_transformer/blob/main/vit_jax/configs/models.py
    """
    assert 'body_parts' in kwargs
    kwargs['body_parts'] = None
    return tamevfm_dino_vit_tiny(**kwargs)


def modality_vfm_dino_vit_small(**kwargs):
    assert 'body_parts' in kwargs
    kwargs['body_parts'] = None
    return tamevfm_dino_vit_small(**kwargs)


def modality_vfm_dino_vit_base(**kwargs):
    assert 'body_parts' in kwargs
    kwargs['body_parts'] = None
    return tamevfm_dino_vit_base(**kwargs)


def modality_vfm_dino_vit_large(**kwargs):
    assert 'body_parts' in kwargs
    kwargs['body_parts'] = None
    return tamevfm_dino_vit_large(**kwargs)


def modality_vfm_dino_vit_huge(**kwargs):
    assert 'body_parts' in kwargs
    kwargs['body_parts'] = None
    return tamevfm_dino_vit_huge(**kwargs)


def modality_vfm_dino_vit_giant2(**kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    assert 'body_parts' in kwargs
    kwargs['body_parts'] = None
    return tamevfm_dino_vit_giant2(**kwargs)
