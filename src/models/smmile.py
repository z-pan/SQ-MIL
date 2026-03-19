"""
SMMILe — Superpatch-based Measurable Multiple Instance Learning model.

Combines:
  - NIC feature rearrangement (3×3 conv)
  - Per-category gated attention instance detector
  - Instance Dropout (InD)
  - Delocalized Instance Sampling (InS) via SLIC superpatches
  - Stage 2: instance refinement + MRF (loaded separately in Trainer)

Reference: Gao et al., Nature Cancer 2025.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nic import NICLayer
from .attention import GatedAttention
from .instance_refinement import InstanceRefinement


class SMMILe(nn.Module):
    """Full SMMILe model for multiclass WSI classification.

    Args:
        embedding_dim:   Encoder output dimensionality (512 Conch / 1024 ResNet-50).
        n_classes:       Number of cancer subtypes.
        nic_out_channels: NIC conv output channels.
        nic_kernel_size: NIC conv kernel (3 for ovarian, 1 for Camelyon16).
        attn_hidden_dim: Gated attention hidden dim.
        attn_dropout:    Whether to use dropout in attention.
        attn_dropout_rate: Attention dropout probability.
        ind_drop_rate:   Instance Dropout rate (fraction of top instances dropped).
        n_refinement_layers: Number of Stage 2 refinement layers (0 disables Stage 2).
        ref_hidden_dim:  Refinement layer hidden dim.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        n_classes: int = 5,
        nic_out_channels: int = 256,
        nic_kernel_size: int = 3,
        attn_hidden_dim: int = 256,
        attn_dropout: bool = True,
        attn_dropout_rate: float = 0.25,
        ind_drop_rate: float = 0.5,
        n_refinement_layers: int = 0,
        ref_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.ind_drop_rate = ind_drop_rate

        # Stage 1 components
        self.nic = NICLayer(
            in_channels=embedding_dim,
            out_channels=nic_out_channels,
            kernel_size=nic_kernel_size,
        )
        self.attention = GatedAttention(
            in_dim=nic_out_channels,
            hidden_dim=attn_hidden_dim,
            n_classes=n_classes,
            dropout=attn_dropout,
            dropout_rate=attn_dropout_rate,
        )
        # Per-category instance classifier (bag label via attention-pooled features)
        self.instance_classifier = nn.Linear(nic_out_channels, n_classes)

        # Stage 2 components (optional; instantiated when n_refinement_layers > 0)
        self.refinement: InstanceRefinement | None = None
        if n_refinement_layers > 0:
            self.refinement = InstanceRefinement(
                in_dim=nic_out_channels,
                hidden_dim=ref_hidden_dim,
                n_classes=n_classes,
                n_layers=n_refinement_layers,
            )

    # ------------------------------------------------------------------
    # Instance Dropout (InD)
    # ------------------------------------------------------------------
    def _instance_dropout(self, A: torch.Tensor) -> torch.Tensor:
        """Zero out top-scoring instances stochastically during training.

        Args:
            A: Attention weights (n_classes, N).

        Returns:
            Masked attention weights (n_classes, N).
        """
        if not self.training or self.ind_drop_rate <= 0.0:
            return A
        n_drop = max(1, int(A.shape[1] * self.ind_drop_rate))
        # Drop the same set of top instances for all categories for simplicity.
        top_mean = A.mean(dim=0)  # (N,)
        _, top_idx = top_mean.topk(n_drop)
        mask = torch.ones_like(A)
        mask[:, top_idx] = 0.0
        return A * mask

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        superpixels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x:           Patch embeddings (N, embedding_dim).
            superpixels: Superpixel label map (N,) for InS / MRF. Optional.

        Returns:
            Dictionary with keys:
              - 'bag_logits':   (n_classes,) — bag-level classification logits
              - 'attn_raw':     (n_classes, N) — raw attention scores
              - 'attn':         (n_classes, N) — softmax attention weights
              - 'inst_features':(N, nic_out_channels) — NIC features
              - 'ref_logits':   (N, n_classes+1) — refinement logits [Stage 2 only]
        """
        # NIC feature extraction
        h = self.nic(x)  # (N, nic_out_channels)

        # Gated attention
        A_raw, A = self.attention(h)  # both (n_classes, N)

        # Instance Dropout
        A_drop = self._instance_dropout(A)

        # Bag aggregation: weighted sum of features per category
        # M: (n_classes, nic_out_channels)
        M = torch.mm(A_drop, h)

        # Bag-level logits: dot product of aggregated features and per-class weights
        bag_logits = self.instance_classifier(M).diagonal()  # (n_classes,)

        out: dict[str, torch.Tensor] = {
            "bag_logits": bag_logits,
            "attn_raw": A_raw,
            "attn": A,
            "inst_features": h,
        }

        # Stage 2 refinement
        if self.refinement is not None:
            _, ref_logits = self.refinement(h)  # (N, n_classes+1)
            out["ref_logits"] = ref_logits

        return out
