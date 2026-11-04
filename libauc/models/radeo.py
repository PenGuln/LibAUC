import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from .builder_vfm import build_model


## List of supported modalities and body parts
MODALITIES = [
    "X-Ray",
    "CT",
    "Microscopy",
    "Endoscopy",
    "Fundus",
    "Dermoscopy",
    "OCTA",
    "Ultrasound",
    "MRI"
]
BODY_PARTS = [
    "Chest",
    "Abdomen",
    "Skin",
    "Eye",
    "N/A"
]

## Some need OCT; must stay appended at the end so existing indices
MODALITY_95M = MODALITIES + ["OCT"]


def get_modalities(train_data: str):
    """Select the modality table for a run based on its train_data string. Done to keep behavior same.
    The filtered 60M and 95M unfiltered need extra OCT.
    """
    if "OmniMed-95M" in train_data or "Filtered-60M" in train_data:
        return MODALITY_95M
    return MODALITIES

def clean_none(arg):
    if arg is None:
        return None
    arg_l = arg.strip().lower()
    if arg_l == 'none' or arg_l == '':
        return None
    elif arg_l == 'true':
        return True
    elif arg_l == 'false':
        return False
    return arg


## get model function that returns the appropriate model based on args.style
def get_model(args) -> nn.Module:
    kwargs_model = dict(
        model_flag=args.style,
        n_class = args.dim,
        enable_finetune=True,
        pretrain_model="imageNet", #"None"/"imageNet"
        modalities=get_modalities(getattr(args, 'train_data', '') or ''),
        body_parts=BODY_PARTS,
        ada_ln_zero_init=args.ada_ln_zero_init,
        attention_lora_rank=args.attention_lora_rank,
        mlp_head=args.mlp_head,
        mlp_head_dim=args.mlp_head_dim,
        conditioned_head=args.conditioned_head,
    )
    args.patch_embed_type = clean_none(args.patch_embed_type)
    if args.patch_embed_type is not None:
        kwargs_model['embed_layer'] = args.patch_embed_type
    args.attention_layer = clean_none(args.attention_layer)
    if args.attention_layer is not None:
        kwargs_model['attn_class'] = args.attention_layer
    return build_model(**kwargs_model)


def sanitize_checkpoint(ckpt):
    ckpt = {k.replace('module.', '').replace('_orig_mod.', ''):v for k,v in ckpt.items()}
    return {k.replace('base_encoder.', ''):v for k,v in ckpt.items() if k.startswith('base_encoder')}


class RADEOEncoder(nn.Module):
    def __init__(self, model, modality_id, body_part_id):
        super().__init__()
        self.model = model
        self.modality_id = modality_id
        self.body_part_id = body_part_id

    def forward(self, images):
        batch_size = images.shape[0]
        modality = torch.full(
            (batch_size,), self.modality_id, dtype=torch.long, device=images.device
        )
        body_part = torch.full(
            (batch_size,), self.body_part_id, dtype=torch.long, device=images.device
        )
        return self.model(images, modality=modality, body_part=body_part)


def load_radeo(pretrained_path, config_path=None, modality="MRI", body_part="N/A"):
    args = argparse.Namespace(model=str(pretrained_path))
    if os.path.exists(args.model) and args.model.endswith(".pth"):
        latest_ckpt_path = os.path.join(args.model, "checkpoint.pth.tar")
        args.model = str(Path(args.model).parent)
    elif os.path.exists(args.model) and args.model.endswith(".pth.tar"):
        latest_ckpt_path = args.model
        args.model = str(Path(args.model).parent.parent)
    else:
        checkpoints = [
            f for f in os.listdir(args.model)
            if f.startswith("checkpoint_") and f.endswith(".pth")
        ]
        latest_ckpt = sorted(checkpoints)[-1]
        latest_ckpt_path = os.path.join(args.model, latest_ckpt, "checkpoint.pth.tar")

    checkpoint = torch.load(latest_ckpt_path, map_location="cpu")
    checkpoint = sanitize_checkpoint(checkpoint['state_dict'])

    if config_path is not None:
        config_path = str(config_path)
        with open(config_path, "r") as f:
            pretrained_args = json.load(f) if config_path.endswith(".json") else yaml.safe_load(f)
    elif os.path.exists(os.path.join(args.model, "config.yaml")):
        with open(os.path.join(args.model, "config.yaml"), "r") as f:
            pretrained_args = yaml.safe_load(f)
    elif os.path.exists(os.path.join(args.model, "config.json")):
        with open(os.path.join(args.model, "config.json"), "r") as f:
            pretrained_args = json.load(f)
    else:
        raise ValueError(f"No config file found in {args.model}")

    default_training_args = {
        "dim": 128,
        "train_data": "",
        "ada_ln_zero_init": False,
        "attention_lora_rank": 0,
        "mlp_head": False,
        "mlp_head_dim": 2048,
        "conditioned_head": False,
        "patch_embed_type": None,
        "attention_layer": None,
    }
    default_training_args.update(pretrained_args)
    pretrained_args = argparse.Namespace(**default_training_args)
    del default_training_args

    model = get_model(pretrained_args)
    model.load_state_dict(checkpoint, strict=True)

    model.head = nn.Identity()
    if hasattr(model, "head_is_conditioned"):
        model.head_is_conditioned = False
    model.mask_token.requires_grad = False

    modalities = get_modalities(getattr(pretrained_args, 'train_data', '') or '')
    if modality not in modalities:
        raise ValueError(f"Unknown modality: {modality}")
    if body_part not in BODY_PARTS:
        raise ValueError(f"Unknown body part: {body_part}")

    encoder = RADEOEncoder(
        model,
        modality_id=modalities.index(modality),
        body_part_id=BODY_PARTS.index(body_part),
    )
    return encoder, model.embed_dim


__all__ = ["RADEOEncoder", "load_radeo"]
