"""
NIC (Neural Image Compression) convolutional layer.

Adapts a flat bag of patch embeddings into a spatially-aware feature
representation by rearranging the K instances into the nearest square 2-D grid
and applying a standard Conv2d → BN → ReLU block.

The key insight (SMMILe §3.1) is that a 3×3 convolutional kernel on this
pseudo-image lets each instance aggregate context from spatially-adjacent
patches, which is important for capturing local tissue structure without
requiring explicit spatial coordinates at this stage.

Paper equation (informal)
--------------------------
  H = ReLU( BN( Conv_{3×3}( rearrange(X) ) ) )

where X ∈ R^{K×D_in} is the bag of pre-extracted embeddings and the
rearrangement maps K instances to a ⌈√K⌉ × ⌈√K⌉ grid (zero-padded if
K is not a perfect square), and H ∈ R^{K×D_out} is the output.

References
----------
* Gao et al. (2025) — SMMILe, Nature Cancer (§3.1, NIC-based feature
  rearrangement with 3×3 conv, 128 output channels for binary tasks
  and 256 for multiclass — see config ovarian_conch_s1.yaml).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NICLayer(nn.Module):
    """Rearrange a bag of patch embeddings into a 2-D grid and apply Conv2d.

    Flow
    ----
    1. Pad K instances to a square of side ⌈√K⌉ (zero-pad along the
       *patch* dimension, not the feature dimension).
    2. Reshape to ``(1, D_in, side, side)`` — a single-image batch.
    3. Apply ``Conv2d(D_in, D_out, kernel_size) → BN → ReLU``.
    4. Reshape output back to ``(K, D_out)`` (remove the padding rows).

    Parameters
    ----------
    in_channels:
        Embedding dimensionality from the upstream encoder
        (512 for Conch, 1024 for ResNet-50 third-stage features).
    out_channels:
        Number of output feature channels (256 per config; 128 for binary tasks).
    kernel_size:
        Convolution kernel size.  Use **3** for ovarian cancer (multiclass)
        as specified in CLAUDE.md; use 1 only for Camelyon16.
    padding_mode:
        How to pad the 2-D feature map at the borders.
        ``'zeros'`` (default) is sufficient; ``'reflect'`` can reduce border
        artefacts but is slower.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
            bias=False,   # BN provides its own bias
        )
        # BatchNorm2d is applied across the (H, W) spatial extent of the
        # pseudo-image; for bags with ≥4 patches (side ≥ 2) this is stable.
        # For bags with a single patch (side=1) BatchNorm degenerates to an
        # identity — the guard below handles that edge case.
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply NIC convolution to a bag of patch embeddings.

        Parameters
        ----------
        x : (K, in_channels)
            All K patch embeddings for a single WSI (one bag).

        Returns
        -------
        out : (K, out_channels)
            Spatially-contextualised instance features H.
        """
        K, C = x.shape
        side = math.ceil(math.sqrt(max(K, 1)))  # grid side length ≥ 1

        # --- 1. Zero-pad along patch dimension to fill the square grid
        pad_len = side * side - K
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))   # pad at end: (side*side, C)

        # --- 2. Reshape to (1, C, side, side) pseudo-image
        feat_map = (
            x.view(1, side, side, C)            # (1, H, W, C)
            .permute(0, 3, 1, 2)                # (1, C, H, W)
            .contiguous()
        )

        # --- 3. Conv2d → BN → ReLU
        if side == 1:
            # BatchNorm2d needs >1 element per channel when training;
            # skip norm for degenerate single-patch bags.
            out = self.act(self.conv(feat_map))  # (1, out_channels, 1, 1)
        else:
            out = self.act(self.norm(self.conv(feat_map)))  # (1, D_out, H, W)

        # --- 4. Flatten back to (K, out_channels); discard padded rows
        out = (
            out.permute(0, 2, 3, 1)             # (1, H, W, D_out)
            .reshape(side * side, self.out_channels)
        )
        out = out[:K]                            # (K, out_channels)

        return out
