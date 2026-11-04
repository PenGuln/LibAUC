from .layers import *
from .blocks import ConditioningEmbedder
# our conditioned models for classification
from .classifier import tamevfm_dino_vit_tiny, tamevfm_dino_vit_small, tamevfm_dino_vit_base, tamevfm_dino_vit_large, tamevfm_dino_vit_huge, tamevfm_dino_vit_giant2, modality_vfm_dino_vit_tiny, modality_vfm_dino_vit_small, modality_vfm_dino_vit_base, modality_vfm_dino_vit_large, modality_vfm_dino_vit_huge, modality_vfm_dino_vit_giant2

DEFAULT_DINOV2_KWARGS = dict(
    patch_size=16,
    init_values=1.0e-05,
    block_chunks=4,
    drop_path_rate=0.3,
    drop_path_uniform=True,
)

