"""
Gated attention mechanism for MIL bag aggregation (per-category).

Implements the instance detector described in SMMILe §3.2, which extends the
standard ABMIL gated attention (Ilse et al. 2018) to produce *per-category*
attention weights.  Each of the C categories has an independent scoring head W_c
applied to a shared gated projection, so the model can identify category-specific
discriminative instances.

Paper equations
---------------
Let h_k ∈ R^D be the NIC feature of instance k.

  v_k = tanh(V · h_k)              V ∈ R^{L×D}              (tanh branch)
  u_k = σ(U · h_k)                 U ∈ R^{L×D}              (sigmoid gate)
  g_k = v_k ⊙ u_k                                            (gated feature)

  e_{c,k} = W_c^T · g_k            W_c ∈ R^L                (raw score, per cat.)
  a_{c,k} = softmax_k(e_{c,k})     ∑_k a_{c,k} = 1          (attention weight)

  bag representation: M_c = ∑_k a_{c,k} · h_k               (category-c bag repr.)

References
----------
* Ilse et al. (2018) — "Attention-based Deep MIL" (original gated attention)
* Gao et al. (2025) — SMMILe, Nature Cancer — per-category extension (§3.2)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttention(nn.Module):
    """Per-category gated attention detector.

    Produces attention weights a_{c,k} (Eq. in SMMILe §3.2) for every
    (category c, instance k) pair.

    Parameters
    ----------
    in_dim:
        Dimensionality of the input instance features (NIC output channels).
    hidden_dim:
        Width L of the shared tanh / sigmoid projection layers (V and U).
        Paper and configs typically use 256 for this project.
    n_classes:
        Number of cancer subtypes C.  One independent scoring head W_c per class.
    dropout:
        Whether to apply dropout to instance features before attention.
    dropout_rate:
        Dropout probability.
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

        # Shared projections — applied to every instance h_k
        # V: tanh branch   (R^D → R^L)
        # U: sigmoid gate  (R^D → R^L)
        self.V = nn.Linear(in_dim, hidden_dim, bias=True)
        self.U = nn.Linear(in_dim, hidden_dim, bias=True)

        # Per-category scoring heads W_c : R^L → R^1,  c = 0..C-1
        # Using a ModuleList so each W_c has independent parameters.
        self.W = nn.ModuleList(
            [nn.Linear(hidden_dim, 1, bias=True) for _ in range(n_classes)]
        )

    # ------------------------------------------------------------------
    def forward(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-category attention weights for a bag of instances.

        Parameters
        ----------
        h : (K, in_dim)
            Instance features for all K patches in one WSI (one bag).

        Returns
        -------
        A_raw : (n_classes, K)
            Raw (unnormalised) attention logits e_{c,k}.
        A : (n_classes, K)
            Softmax-normalised attention weights a_{c,k}.
            For each class c:  sum_k A[c, k] == 1.
        """
        h = self.dropout(h)                       # (K, D)

        v = torch.tanh(self.V(h))                 # (K, L)  — tanh branch
        u = torch.sigmoid(self.U(h))              # (K, L)  — gate branch
        g = v * u                                 # (K, L)  — gated feature g_k

        # Stack per-category raw scores: e_{c,k} = W_c^T g_k
        # Each W[c] maps (K, L) → (K, 1); squeeze and stack → (C, K)
        A_raw = torch.stack(
            [self.W[c](g).squeeze(-1) for c in range(self.n_classes)],
            dim=0,
        )  # (n_classes, K)

        # Softmax over instances (dim=-1) per category
        A = F.softmax(A_raw, dim=-1)              # (n_classes, K)

        return A_raw, A
