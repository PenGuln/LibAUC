import logging

import torch
import torch.nn as nn

# Conditioning embedder
# Here the plan is to create the conditioning vector for modality | body part combination

class ConditioningEmbedder(nn.Module):
    """
    Learnable conditioning embedding for (modality, body-part) pairs.
    Each unique combination gets its own embedding vector, which is then
    projected to a fixed dimension (e.g., 128).
    """

    def __init__(
            self, 
            num_combinations,
            embed_dim=128
        ):
        super().__init__()
        self.num_combinations = num_combinations

        ## Embedding for each (modality, bodypart) pair
        self.embedding = nn.Embedding(self.num_combinations, embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.embedding.weight, std=0.02)

    def forward(self, cond_idx):
        """
        Returns:
            torch.Tensor: conditioning embedding of shape (B, embed_dim)
        """
        # Lookup embeddings
        cond = self.embedding(cond_idx)
        return cond


# Conditioning embedder for modality only (without body part)

# class ModalityConditioningEmbedder(nn.Module):
#     """
#     Learnable conditioning embedding for (modality).
#     Each unique combination gets its own embedding vector, which is then
#     projected to a fixed dimension (e.g., 128).
#     """

#     def __init__(self, modalities, embed_dim=128):
#         super().__init__()
#         self.modalities = modalities
#         self.num_modalities = len(modalities)
#         self.num_combinations = self.num_modalities
#         # Embedding for each modality
#         self.embedding = nn.Embedding(self.num_combinations, embed_dim)
#         self._init_weights()

#     def _init_weights(self):
#         nn.init.trunc_normal_(self.embedding.weight, std=0.02)


#     def forward(self, modality):
#         """
#         Returns:
#             torch.Tensor: conditioning embedding of shape (B, embed_dim)
#         """
#         # Lookup embeddings
#         cond = self.embedding(modality)
#         return cond