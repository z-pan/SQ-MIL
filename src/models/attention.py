"""
Gated attention mechanism for MIL bag aggregation.

Implements the per-category gated attention detector described in SMMILe §3.2.
Each category has an independent attention branch so the model can identify
category-specific discriminative instances.

Reference: Ilse et al. (2018) — Attention-based Deep MIL.
           SMMILe — per-category extension with gating.
"""

import torch
import torch.nn as nn


class GatedAttention(nn.Module):
    """Gated attention module producing per-category instance weights.

    Args:
        in_dim:     Input feature dimensionality (NIC output channels).
        hidden_dim: Dimensionality of the attention hidden layer.
        n_classes:  Number of categories (one attention branch each).
        dropout:    Whether to apply dropout before attention computation.
        dropout_rate: Dropout probability.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        n_classes: int = 5,
        dropout: bool = True,
        dropout_rate: float = 0.25,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes

        self.dropout = nn.Dropout(p=dropout_rate) if dropout else nn.Identity()

        # Shared feature projection
        self.V = nn.Linear(in_dim, hidden_dim)   # tanh branch
        self.U = nn.Linear(in_dim, hidden_dim)   # sigmoid gate branch

        # Per-category attention scoring
        self.W = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(n_classes)]
        )

    def forward(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-category attention weights over the bag.

        Args:
            h: Instance features of shape (N, in_dim).

        Returns:
            A_raw: Raw attention logits, shape (n_classes, N).
            A:     Softmax-normalised attention weights, shape (n_classes, N).
        """
        h = self.dropout(h)

        v = torch.tanh(self.V(h))       # (N, hidden_dim)
        u = torch.sigmoid(self.U(h))    # (N, hidden_dim)
        gated = v * u                   # (N, hidden_dim)

        A_raw = torch.stack(
            [self.W[c](gated).squeeze(-1) for c in range(self.n_classes)],
            dim=0,
        )  # (n_classes, N)

        A = torch.softmax(A_raw, dim=-1)  # (n_classes, N)
        return A_raw, A
