import torch
import torch.nn as nn


class SharedConditionedLinear(nn.Module):
    """
    Head/Projection layer with shared + modality specific weights, same idea as Patch Embedding
    """

    def __init__(
        self,
        num_combinations: int,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.num_combinations = num_combinations
        self.in_features = in_features
        self.out_features = out_features

        self.shared_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.weights = nn.Parameter(torch.empty(num_combinations, out_features, in_features))
        self.biases = nn.Parameter(torch.zeros(num_combinations, out_features))

        self._init_weights()

    def _init_weights(self):
        w = self.weights.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.xavier_uniform_(self.shared_weight.data)
        nn.init.constant_(self.biases, 0)

    def forward(self, x: torch.Tensor, cond_idx: torch.Tensor) -> torch.Tensor:
        w_s = self.weights[cond_idx] + self.shared_weight  # (B, out_features, in_features)
        b_s = self.biases[cond_idx]  # (B, out_features)
        return torch.einsum('boi,bi->bo', w_s, x) + b_s


class SharedConditionedMlpHead(nn.Module):
    """
    Mlp head (Linear -> LayerNorm -> ReLU -> Linear) with shared + per-combination
    specific weights on both linear layers.
    """

    def __init__(
        self,
        num_combinations: int,
        in_features: int,
        hidden_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.fc1 = SharedConditionedLinear(num_combinations, in_features, hidden_features)
        self.norm = nn.LayerNorm(hidden_features)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = SharedConditionedLinear(num_combinations, hidden_features, out_features)

    def forward(self, x: torch.Tensor, cond_idx: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x, cond_idx)
        x = self.norm(x)
        x = self.act(x)
        x = self.fc2(x, cond_idx)
        return x
