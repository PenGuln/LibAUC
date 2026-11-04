
import os
import numpy as np
import torch
from functools import partial

from timm.models.layers import trunc_normal_
from .models_vit import vit_tiny_patch16, vit_small_patch16, vit_base_patch16, vit_large_patch16, vit_huge_patch14
from . import tamevfm


AVAILABLE_MODELS = {
    # ViT models
    "vit_tiny_patch16": vit_tiny_patch16,
    "vit_small_patch16": vit_small_patch16,
    "vit_base_patch16": vit_base_patch16,
    "vit_large_patch16": vit_large_patch16,
    "vit_huge_patch14": vit_huge_patch14,
    # tamevfm models
    "tame_cvit_tiny": tamevfm.tamevfm_dino_vit_tiny,
    "tame_cvit_small": tamevfm.tamevfm_dino_vit_small,
    "tame_cvit_base": tamevfm.tamevfm_dino_vit_base,
    "tame_cvit_base_reg4": partial(tamevfm.tamevfm_dino_vit_base, num_register_tokens=4),
    "tame_cvit_large": tamevfm.tamevfm_dino_vit_large,
    "tame_cvit_huge": tamevfm.tamevfm_dino_vit_huge,
    # modality-vfm models
    "modality_vfm_tiny": tamevfm.modality_vfm_dino_vit_tiny,
    "modality_vfm_small": tamevfm.modality_vfm_dino_vit_small,
    "modality_vfm_base": tamevfm.modality_vfm_dino_vit_base,
    "modality_vfm_large": tamevfm.modality_vfm_dino_vit_large,
    "modality_vfm_huge": tamevfm.modality_vfm_dino_vit_huge,
}

def _init_head_weights(head, std=0.01):
    """trunc_normal_ init for a classification/projection head that may be a
    bare nn.Linear or an nn.Sequential (e.g. an mlp_head)."""
    if isinstance(head, torch.nn.Linear):
        trunc_normal_(head.weight, std=std)
    elif isinstance(head, torch.nn.Sequential):
        for m in head:
            if isinstance(m, torch.nn.Linear):
                trunc_normal_(m.weight, std=std)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    # omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

def build_vit_model(model_flag, n_classes, global_pool, pretrain_model, enable_finetune, **kwargs):

    use_imagenet_pretrain = False
    if pretrain_model.lower() == "imagenet":
        assert model_flag in ['vit_base_patch16', 'vit_large_patch16']
        use_imagenet_pretrain = True

    # add default kwargs
    DEFAULT_DINOV2_KWARGS = dict(
    init_values=1.0e-05,
    drop_path_rate=0.001,)

    # add default kwargs : Same as tamevfm conditioned models
    for k, v in DEFAULT_DINOV2_KWARGS.items():
        if k not in kwargs:
            kwargs[k] = v

    model = AVAILABLE_MODELS[model_flag](
        n_classes=n_classes,
        global_pool=global_pool,
        pretrained=use_imagenet_pretrain,
        **kwargs,
    )

    # load pretrain checkpoint
    if os.path.isfile(pretrain_model):
        checkpoint = torch.load(pretrain_model, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % pretrain_model)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        if global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer: following MoCo v3
        _init_head_weights(model.head)
    else:
        if not use_imagenet_pretrain:
            print("Pretrain model checkpoint file does not exist!")
            print("Initialize model with random weights.")

    # employ linear probing by default
    if enable_finetune:
        print("Enable finetune mode: all parameters are trainable.")
    else:
        print("Enable linear probing mode: only head parameters are trainable.")
        # for linear prob only
        # hack: revise model's head with BN
        model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
        # freeze all but the head
        for _, p in model.named_parameters():
            p.requires_grad = False
        for _, p in model.head.named_parameters():
            p.requires_grad = True

    return model


def build_tamevfm_model(model_flag, n_classes, global_pool, pretrain_model, enable_finetune, modalities, body_parts, **kwargs):
    # check the needs of load imagenet model
    use_imagenet_pretrain = False
    if pretrain_model.lower() == "imagenet":
        use_imagenet_pretrain = True

    # add default kwargs
    DEFAULT_DINOV2_KWARGS = dict(
    patch_size=16,
    init_values=1.0e-05,
    block_chunks=4,
    drop_path_rate=0.001, #to use drop path we reduce it
    drop_path_uniform=True,)

    # add default kwargs
    assert 'init_values' not in kwargs
    for k, v in DEFAULT_DINOV2_KWARGS.items():
        if k not in kwargs:
            kwargs[k] = v
    kwargs.pop('init_values')

    # patch_size is 16 by default, except for huge model
    if model_flag.endswith('_huge'):
        kwargs['patch_size'] = 14
        kwargs['drop_path_rate'] = 0.001
        kwargs['block_chunks'] = 4

    model = AVAILABLE_MODELS[model_flag](
        num_classes=n_classes,
        global_pool=global_pool,
        modalities=modalities,
        body_parts=body_parts,
        **kwargs,
    )

    # load pretrain checkpoint
    if use_imagenet_pretrain:
        import timm, re
        import torch.distributed as dist
        if 'huge' in model_flag:
            timm_model_name = 'vit_huge_patch14_224.orig_in21k'
        elif 'large' in model_flag:
            timm_model_name = 'vit_large_patch16_224'
        else:
            timm_model_name = 'vit_base_patch16_224'
        print(f"Loading ImageNet {timm_model_name} weights from timm...")
        # Serialize the HF hub download to local rank 0 per node to avoid
        # concurrent NFS flock/fchmod failures (OSError 116 stale file handle).
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        is_dist = dist.is_available() and dist.is_initialized()
        if local_rank == 0:
            vit_imagenet = timm.create_model(timm_model_name, pretrained=True)
        if is_dist:
            dist.barrier()
        if local_rank != 0:
            vit_imagenet = timm.create_model(timm_model_name, pretrained=True)
        sd = vit_imagenet.state_dict()

        # drop keys that won't match (different class count, no norm affine in adaLN blocks)
        sd.pop('head.weight', None)
        sd.pop('head.bias', None)
        sd.pop('norm.weight', None)
        sd.pop('norm.bias', None)

        # patch_embed.proj is a single Conv2d in vanilla ViT;
        # FasterConditionedPatchEmbed uses shared_weights + per-combination weights instead
        conv_w = sd.pop('patch_embed.proj.weight')   # [embed_dim, 3, 16, 16]
        sd.pop('patch_embed.proj.bias', None)
        with torch.no_grad():
            model.patch_embed.shared_weights.copy_(conv_w)
            model.patch_embed.weights.copy_(
                conv_w.unsqueeze(0).expand_as(model.patch_embed.weights).clone()
            )

        # timm: blocks.{i}.* (flat, 0-11)
        # ConditionedViT with block_chunks=4: blocks.{chunk}.{i}.* where chunk = i // chunksize
        # e.g. depth=12, block_chunks=4 -> chunksize=3: block 0->blocks.0.0, block 3->blocks.1.3
        depth = model.n_blocks
        block_chunks = kwargs.get('block_chunks', 4)
        chunksize = depth // block_chunks
        remapped_sd = {}
        for k, v in sd.items():
            m = re.match(r'^blocks\.(\d+)\.(.+)$', k)
            if m:
                i = int(m.group(1))
                rest = m.group(2)
                chunk = i // chunksize
                new_key = f'blocks.{chunk}.{i}.{rest}'
                remapped_sd[new_key] = v
            else:
                remapped_sd[k] = v

        msg = model.load_state_dict(remapped_sd, strict=False)
        print("ImageNet ViT weights loaded.")
        print("  Missing (conditioning-specific, random init):", msg.missing_keys)
        print("  Unexpected (imagenet-only, skipped):", msg.unexpected_keys)

        _init_head_weights(model.head)

    elif os.path.isfile(pretrain_model):
        checkpoint = torch.load(pretrain_model, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % pretrain_model)
        checkpoint_model = checkpoint['teacher']
        # remove `module.` prefix
        checkpoint_model = {k.replace("module.", ""): v for k, v in checkpoint_model.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        if global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer: following MoCo v3
        _init_head_weights(model.head)
    else:
        print("Pretrain model checkpoint file does not exist!")
        print("Initialize model with random weights.")

    # employ linear probing by default
    if enable_finetune:
        print("Enable finetune mode: all parameters are trainable.")
    else:
        print("Enable linear probing mode: only head parameters are trainable.")
        # hack: revise model's head with BN
        model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
        # freeze all but the head
        for _, p in model.named_parameters():
            p.requires_grad = False
        for _, p in model.head.named_parameters():
            p.requires_grad = True

    return model

def build_model(model_flag, n_class, pretrain_model, enable_finetune, modalities, body_parts, **kwargs):
    assert model_flag in AVAILABLE_MODELS, f"Model {model_flag} is not available. Please choose from {list(AVAILABLE_MODELS.keys())}."

    if "tame" in model_flag or "modality_vfm" in model_flag:
        model = build_tamevfm_model(
            model_flag=model_flag,
            n_classes=n_class,
            global_pool=kwargs.get('global_pool', False),
            pretrain_model=pretrain_model,
            enable_finetune=enable_finetune,
            modalities=modalities,
            body_parts=body_parts,
            **kwargs,
        )
    elif "vit" in model_flag:
        model = build_vit_model(
            model_flag=model_flag,
            n_classes=n_class,
            global_pool=kwargs.get('global_pool', False),
            pretrain_model=pretrain_model,
            enable_finetune=enable_finetune,
            **kwargs,
        )
    else:
        raise NotImplementedError


    return model
