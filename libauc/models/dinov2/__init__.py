from .layers import *

# # original dino models
# from .vision_transformer import vit_small, vit_base, vit_large, vit_giant2

# our modificated models for classification
from .classifier import dino_vit_tiny, dino_vit_small, dino_vit_base, dino_vit_large, dino_vit_huge, dino_vit_giant2


DEFAULT_DINOV2_KWARGS = dict(
    patch_size=16,
    init_values=1.0e-05,
    block_chunks=4,
    drop_path_rate=0.3,
    drop_path_uniform=True,
)

