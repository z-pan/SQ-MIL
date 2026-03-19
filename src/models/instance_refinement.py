"""
Instance refinement network (Stage 2).

Stacks N linear layers that progressively refine patch-level predictions
using pseudo-labels generated from Stage 1 output. The final layer produces
(C+1)-dim softmax scores used for heatmap generation.

Reference: SMMILe §3.3 — Instance Refinement with progressive pseudo-labeling.
"""

import torch
import torch.nn as nn


class InstanceRefinement(nn.Module):
    """N stacked linear refinement layers.

    Args:
        in_dim:    Dimensionality of the input instance features.
        hidden_dim: Hidden layer width.
        n_classes: Number of cancer subtypes (C).
        n_layers:  Number of refinement layers (N; paper uses 3).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        n_classes: int = 5,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.n_classes = n_classes
        # Output dim is C+1: C cancer subtypes + 1 background/normal
        out_dim = n_classes + 1

        layers: list[nn.Module] = []
        for i in range(n_layers):
            dim_in = in_dim if i == 0 else hidden_dim
            layers.append(
                nn.Sequential(
                    nn.Linear(dim_in, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.25),
                )
            )
        self.layers = nn.ModuleList(layers)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(
        self, h: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Run N refinement layers and return intermediate + final predictions.

        Args:
            h: Instance features of shape (N, in_dim).

        Returns:
            intermediates: List of N intermediate hidden tensors (N_inst, hidden_dim).
            logits:        Final logits of shape (N_inst, C+1).
        """
        intermediates: list[torch.Tensor] = []
        x = h
        for layer in self.layers:
            x = layer(x)
            intermediates.append(x)

        logits = self.classifier(x)  # (N_inst, C+1)
        return intermediates, logits
