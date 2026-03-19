"""
NIC (Neural Image Compression) convolutional layer.

Adapts patch embeddings from a pretrained encoder into a spatially-aware
feature representation using a 2D convolution applied to a square-rearranged
feature map.

Reference: SMMILe §3.1 — NIC-based feature rearrangement.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NICLayer(nn.Module):
    """Rearranges a 1-D bag of patch embeddings into a 2-D feature map and
    applies a convolutional layer, then flattens back to (N, out_channels).

    Args:
        in_channels:  Embedding dimensionality from the upstream encoder
                      (512 for Conch, 1024 for ResNet-50).
        out_channels: Number of output feature channels.
        kernel_size:  Convolution kernel size. Use 3 for ovarian/multiclass;
                      1 for Camelyon16 binary task.
        padding_mode: Padding strategy ('zeros' or 'reflect').
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
            bias=True,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Patch embedding bag of shape (N, in_channels), where N is the
               number of patches in the WSI.

        Returns:
            Tensor of shape (N, out_channels) with spatially-contextualised
            features.
        """
        N, C = x.shape
        # Arrange N patches into the nearest square grid (pad if needed).
        side = math.ceil(math.sqrt(N))
        pad_len = side * side - N
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))  # pad along patch dimension

        # (side*side, C) → (1, C, side, side)
        feat_map = x.view(1, side, side, C).permute(0, 3, 1, 2).contiguous()

        out = self.act(self.norm(self.conv(feat_map)))  # (1, out_channels, side, side)

        # Flatten back to (side*side, out_channels) and remove padding
        out = out.permute(0, 2, 3, 1).reshape(side * side, self.out_channels)
        out = out[:N]  # remove padded rows

        return out
