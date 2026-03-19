"""
Heatmap generation for SMMILe ovarian cancer 5-subtype predictions.

Produces per-WSI PNG overlays where each patch is colored by its predicted
subtype. Supports both pyramidal (OpenSlide) and flat (PIL/tifffile) TIFF.

Key adaptations from upstream generate_heatmap.py:
  - Uses *.tif glob (not *.svs)
  - Per-class RGB colors instead of single jet-colormap heatmap
  - Gaussian smoothing on the overlay mask
  - Side-by-side ground-truth comparison when annotations are available
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Color scheme (RGB, uint8)
# ---------------------------------------------------------------------------

SUBTYPE_COLORS: dict[int, tuple[int, int, int]] = {
    0: (230, 25,  75),   # CC   — Red
    1: (60,  180, 75),   # EC   — Green
    2: (255, 225, 25),   # HGSC — Yellow
    3: (0,   130, 200),  # LGSC — Blue
    4: (145, 30,  180),  # MC   — Purple
}

SUBTYPE_NAMES: dict[int, str] = {
    0: "CC",
    1: "EC",
    2: "HGSC",
    3: "LGSC",
    4: "MC",
}


# ---------------------------------------------------------------------------
# Core heatmap builder
# ---------------------------------------------------------------------------

def generate_heatmap(
    wsi_path: str | Path,
    inst_csv: str | Path,
    output_path: str | Path,
    gt_csv: str | Path | None = None,
    thumbnail_max: int = 1024,
    overlay_alpha: float = 0.45,
    gaussian_radius: float = 5.0,
    patch_size: int = 512,
) -> np.ndarray:
    """Generate a subtype-colored overlay heatmap for a single WSI.

    Args:
        wsi_path:        Path to the .tif WSI file.
        inst_csv:        Per-patch prediction CSV with columns:
                         filename, x, y, predicted_class,
                         prob_CC, prob_EC, prob_HGSC, prob_LGSC, prob_MC
        output_path:     Where to save the output PNG.
        gt_csv:          Optional ground-truth annotation CSV (same format).
                         When provided, a side-by-side figure is saved.
        thumbnail_max:   Max dimension of the WSI thumbnail in pixels.
        overlay_alpha:   Transparency of the color overlay [0=transparent, 1=opaque].
        gaussian_radius: Sigma for Gaussian smoothing of the overlay mask.
        patch_size:      Patch size in level-0 pixels (used for scaling).

    Returns:
        RGB numpy array (H, W, 3) of the saved heatmap image.
    """
    from ..datasets.wsi_utils import WSIReader

    wsi_path = Path(wsi_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load thumbnail
    with WSIReader(wsi_path) as reader:
        thumbnail = reader.get_thumbnail(max_size=thumbnail_max)  # (H, W, 3)
        wsi_w, wsi_h = reader.dimensions  # level-0 pixel size

    thumb_h, thumb_w = thumbnail.shape[:2]
    scale_x = thumb_w / wsi_w
    scale_y = thumb_h / wsi_h

    # Load predictions
    pred_df = pd.read_csv(inst_csv)
    overlay = _build_overlay(
        pred_df, thumb_w, thumb_h, scale_x, scale_y, patch_size, gaussian_radius
    )

    result_img = _blend(thumbnail, overlay, overlay_alpha)

    if gt_csv is not None and Path(gt_csv).exists():
        gt_df = pd.read_csv(gt_csv)
        gt_overlay = _build_overlay(
            gt_df, thumb_w, thumb_h, scale_x, scale_y, patch_size, gaussian_radius
        )
        gt_img = _blend(thumbnail, gt_overlay, overlay_alpha)
        # Side-by-side with legend
        legend = _make_legend(thumb_h)
        composite = np.concatenate([thumbnail, result_img, gt_img, legend], axis=1)
    else:
        legend = _make_legend(thumb_h)
        composite = np.concatenate([thumbnail, result_img, legend], axis=1)

    Image.fromarray(composite).save(str(output_path))
    logger.info("Saved heatmap → %s", output_path)
    return composite


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_overlay(
    df: pd.DataFrame,
    thumb_w: int,
    thumb_h: int,
    scale_x: float,
    scale_y: float,
    patch_size: int,
    gaussian_radius: float,
) -> np.ndarray:
    """Build RGBA overlay array from instance predictions."""
    overlay = np.zeros((thumb_h, thumb_w, 4), dtype=np.float32)  # RGBA

    for _, row in df.iterrows():
        x0 = int(row["x"] * scale_x)
        y0 = int(row["y"] * scale_y)
        x1 = min(thumb_w, int((row["x"] + patch_size) * scale_x))
        y1 = min(thumb_h, int((row["y"] + patch_size) * scale_y))

        pred_cls = int(row["predicted_class"])
        color = SUBTYPE_COLORS.get(pred_cls, (128, 128, 128))

        # Use max-probability as alpha weight
        prob_col = f"prob_{SUBTYPE_NAMES.get(pred_cls, str(pred_cls))}"
        alpha_w = float(row.get(prob_col, 1.0))

        overlay[y0:y1, x0:x1, 0] = color[0] / 255.0
        overlay[y0:y1, x0:x1, 1] = color[1] / 255.0
        overlay[y0:y1, x0:x1, 2] = color[2] / 255.0
        overlay[y0:y1, x0:x1, 3] = alpha_w

    # Gaussian smoothing on the alpha channel
    if gaussian_radius > 0:
        overlay[:, :, 3] = gaussian_filter(overlay[:, :, 3], sigma=gaussian_radius)
        # Clip after smoothing
        overlay[:, :, 3] = np.clip(overlay[:, :, 3], 0.0, 1.0)

    return overlay


def _blend(
    thumbnail: np.ndarray,
    overlay: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Alpha-composite overlay onto thumbnail."""
    result = thumbnail.astype(np.float32) / 255.0
    ov_rgb = overlay[:, :, :3]      # (H, W, 3)
    ov_alpha = overlay[:, :, 3:4]   # (H, W, 1) — per-pixel alpha weight

    # Effective alpha = global overlay_alpha × per-pixel confidence
    eff_alpha = alpha * ov_alpha

    result = result * (1 - eff_alpha) + ov_rgb * eff_alpha
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return result


def _make_legend(height: int, cell_h: int = 30, cell_w: int = 120) -> np.ndarray:
    """Create a small legend panel showing class colors."""
    n_classes = len(SUBTYPE_COLORS)
    legend_h = max(height, n_classes * (cell_h + 4) + 10)
    legend = np.ones((legend_h, cell_w, 3), dtype=np.uint8) * 240

    for cls_idx, color in SUBTYPE_COLORS.items():
        y = 10 + cls_idx * (cell_h + 4)
        legend[y : y + cell_h, 5 : 5 + 20] = color
        # Draw label text using OpenCV
        label = SUBTYPE_NAMES.get(cls_idx, str(cls_idx))
        cv2.putText(
            legend,
            label,
            (30, y + cell_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )

    return legend
