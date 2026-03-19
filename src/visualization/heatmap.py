"""
Heatmap generation for SMMILe ovarian cancer 5-subtype predictions.

Produces per-WSI PNG figures where each tissue patch is colored by its
predicted subtype (argmax of instance refinement probabilities).  The alpha
channel of each patch is proportional to prediction confidence so uncertain
regions appear more transparent.

Per-class Gaussian smoothing is applied to create smooth spatial gradients
instead of hard patch boundaries.

Supported CSV formats
---------------------
The trainer (``SMMILeTrainer.evaluate``) outputs:

    slide_id, x, y, predicted_class, prob_CC, prob_EC, prob_HGSC, prob_LGSC, prob_MC

The upstream SMMILe script / alternative pipeline outputs:

    filename, x, y, patch_size, pred_class, prob_CC, prob_EC, prob_HGSC, prob_LGSC, prob_MC

Both are accepted; column names are normalised internally.

Key adaptations from upstream generate_heatmap.py
--------------------------------------------------
- Uses ``WSIReader`` (wsi_utils.py) — supports .tif via OpenSlide / tifffile / PIL
- Per-class RGB colors instead of single jet-colormap probability heatmap
- Per-class Gaussian smoothing (sigma=5) for smooth spatial transitions
- Matplotlib composite figure: original | overlay [| GT overlay], with title +
  bottom legend showing per-class confidence
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Color scheme and class metadata
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

# Reverse lookup: "HGSC" → 2
_NAME_TO_IDX: dict[str, int] = {v: k for k, v in SUBTYPE_NAMES.items()}

# Probability column names in fixed order
_PROB_COLS: list[str] = [f"prob_{SUBTYPE_NAMES[c]}" for c in range(len(SUBTYPE_NAMES))]


# ---------------------------------------------------------------------------
# HeatmapGenerator
# ---------------------------------------------------------------------------

class HeatmapGenerator:
    """Generate subtype-colored spatial prediction heatmaps for WSIs.

    Parameters
    ----------
    subtype_colors:
        Dict mapping class index (0–4) to ``(R, G, B)`` uint8 tuples.
        Defaults to the canonical ovarian-cancer scheme.
    thumbnail_size:
        ``(max_width, max_height)`` bounding box for the WSI thumbnail.
        Aspect ratio is preserved.  Larger values produce higher-resolution
        output but require more memory and time.
    overlay_alpha:
        Global opacity of the colored overlay.  Per-pixel alpha is further
        scaled by prediction confidence, so this is the *maximum* opacity.
    gaussian_sigma:
        Sigma (in thumbnail pixels) for per-class Gaussian smoothing of the
        probability maps.  Higher values produce smoother but less spatially
        precise transitions.

    Examples
    --------
    Single slide::

        gen = HeatmapGenerator()
        df  = pd.read_csv("results/stage2/fold0/inst_predictions_fold0.csv")
        df  = df[df["slide_id"] == "slide_001"].reset_index(drop=True)
        gen.generate("data/wsi/slide_001.tif", df, "results/heatmaps/slide_001.png")

    Batch (all slides in a fold)::

        gen = HeatmapGenerator(thumbnail_size=(1024, 1024))
        gen.generate_batch(
            wsi_dir       = "data/wsi/*.tif",
            predictions_dir = "results/stage2/",
            output_dir    = "results/heatmaps/",
            num_workers   = 8,
        )
    """

    def __init__(
        self,
        subtype_colors: dict[int, tuple[int, int, int]] | None = None,
        thumbnail_size: tuple[int, int] = (2048, 2048),
        overlay_alpha: float = 0.50,
        gaussian_sigma: float = 5.0,
    ) -> None:
        self.subtype_colors  = subtype_colors if subtype_colors is not None else SUBTYPE_COLORS
        self.thumbnail_size  = thumbnail_size
        self.overlay_alpha   = overlay_alpha
        self.gaussian_sigma  = gaussian_sigma
        self.n_classes       = len(self.subtype_colors)

    # ======================================================================
    # Public API
    # ======================================================================

    def generate(
        self,
        wsi_path: str | Path,
        predictions_df: pd.DataFrame,
        output_path: str | Path,
        show_gt: bool = False,
        gt_df: pd.DataFrame | None = None,
    ) -> Path:
        """Generate and save a heatmap PNG for one WSI.

        Parameters
        ----------
        wsi_path:
            Path to the ``.tif`` WSI file.
        predictions_df:
            Instance prediction DataFrame (see module docstring for accepted
            column layouts).
        output_path:
            Destination ``.png`` file.
        show_gt:
            When True and *gt_df* is not None, add a third panel showing the
            ground-truth annotation overlay.
        gt_df:
            Ground-truth annotation DataFrame in the same format as
            *predictions_df*.

        Returns
        -------
        Path to the saved PNG.
        """
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend — safe for workers
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        wsi_path    = Path(wsi_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        slide_id = wsi_path.stem

        # ---- Normalise DataFrames ----------------------------------------
        preds = self._normalize_df(predictions_df)

        # ---- Load WSI thumbnail -----------------------------------------
        thumbnail_pil, (wsi_w, wsi_h) = self._load_thumbnail(wsi_path)
        thumb_arr = np.array(thumbnail_pil)          # (H, W, 3) uint8
        thumb_h, thumb_w = thumb_arr.shape[:2]

        scale_x = thumb_w / wsi_w
        scale_y = thumb_h / wsi_h

        # ---- Infer patch size (majority value, fallback 512) -------------
        patch_size = int(preds["patch_size"].mode().iloc[0]) if "patch_size" in preds.columns else 512

        # ---- Build overlays ---------------------------------------------
        prob_maps = self._build_prob_maps(preds, thumb_w, thumb_h, scale_x, scale_y, patch_size)
        overlay_rgb = self._blend_overlay(thumb_arr, prob_maps)

        gt_overlay_rgb: np.ndarray | None = None
        if show_gt and gt_df is not None:
            gt_preds   = self._normalize_df(gt_df)
            gt_prob_maps = self._build_prob_maps(gt_preds, thumb_w, thumb_h, scale_x, scale_y, patch_size)
            gt_overlay_rgb = self._blend_overlay(thumb_arr, gt_prob_maps)

        # ---- WSI-level prediction from mean patch probabilities ---------
        wsi_pred, wsi_conf, mean_probs = self._wsi_level_prediction(preds)
        pred_name = SUBTYPE_NAMES.get(wsi_pred, str(wsi_pred))

        # ---- Compose matplotlib figure ----------------------------------
        n_panels  = 2 + (1 if gt_overlay_rgb is not None else 0)
        dpi       = 100
        # Reserve 80px at bottom for legend + 50px at top for title
        fig_w_in  = (thumb_w * n_panels) / dpi
        fig_h_in  = (thumb_h + 130) / dpi

        fig, axes = plt.subplots(1, n_panels, figsize=(fig_w_in, fig_h_in), dpi=dpi)
        if n_panels == 2:
            axes = list(axes)
        else:
            axes = list(axes)

        # Panel 1: original thumbnail
        axes[0].imshow(thumb_arr, interpolation="nearest")
        axes[0].set_title("Original WSI", fontsize=11, pad=4)
        axes[0].axis("off")

        # Panel 2: spatial prediction overlay
        axes[1].imshow(overlay_rgb, interpolation="nearest")
        axes[1].set_title("Spatial Prediction", fontsize=11, pad=4)
        axes[1].axis("off")

        # Panel 3 (optional): ground-truth overlay
        if gt_overlay_rgb is not None:
            axes[2].imshow(gt_overlay_rgb, interpolation="nearest")
            axes[2].set_title("Ground Truth", fontsize=11, pad=4)
            axes[2].axis("off")

        # Figure title
        fig.suptitle(
            f"{slide_id}  |  Predicted: {pred_name} ({wsi_conf * 100:.1f}%)",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )

        # Bottom legend — per-class color + name + mean confidence
        legend_handles = []
        for cls_idx in range(self.n_classes):
            color_rgb  = self.subtype_colors.get(cls_idx, (128, 128, 128))
            color_mpl  = tuple(c / 255.0 for c in color_rgb)
            name       = SUBTYPE_NAMES.get(cls_idx, str(cls_idx))
            conf_pct   = mean_probs[cls_idx] * 100 if cls_idx < len(mean_probs) else 0.0
            legend_handles.append(
                mpatches.Patch(color=color_mpl, label=f"{name}: {conf_pct:.1f}%")
            )
        fig.legend(
            handles     = legend_handles,
            loc         = "lower center",
            ncol        = self.n_classes,
            fontsize    = 10,
            framealpha  = 0.9,
            bbox_to_anchor = (0.5, 0.005),
        )

        plt.subplots_adjust(top=0.91, bottom=0.08, left=0.01, right=0.99, wspace=0.03)
        fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info("Saved heatmap → %s", output_path)
        return output_path

    def generate_batch(
        self,
        wsi_dir: str | Path,
        predictions_dir: str | Path,
        output_dir: str | Path,
        num_workers: int = 8,
        gt_dir: str | Path | None = None,
    ) -> dict[str, Path | Exception]:
        """Generate heatmaps for all WSIs that have a matching prediction CSV.

        Parameters
        ----------
        wsi_dir:
            Glob pattern (e.g. ``"data/wsi/*.tif"``) or a directory path.
            All ``.tif`` files found are processed.
        predictions_dir:
            Directory (or its sub-directories) that contain prediction CSVs.
            The method recursively scans for ``*.csv`` files and builds a
            ``{slide_id: DataFrame}`` mapping.  Both formats are accepted:
            - Single-file with ``slide_id`` column (trainer output)
            - Per-slide files named ``{slide_id}*.csv``
        output_dir:
            Directory where output ``.png`` files are saved.
        num_workers:
            Number of parallel threads.
        gt_dir:
            Optional directory with ground-truth annotation CSVs named
            ``{slide_id}_gt.csv``.  When found, a GT panel is added.

        Returns
        -------
        Dict mapping slide_id → saved Path (on success) or Exception (on failure).
        """
        import glob

        # ---- Resolve WSI paths ------------------------------------------
        wsi_dir_s = str(wsi_dir)
        if "*" in wsi_dir_s or "?" in wsi_dir_s:
            wsi_paths = sorted(Path(p) for p in glob.glob(wsi_dir_s))
        else:
            wsi_paths = sorted(Path(wsi_dir_s).glob("*.tif"))

        if not wsi_paths:
            logger.warning("No .tif files found matching: %s", wsi_dir_s)
            return {}
        logger.info("Batch mode: %d WSI files found.", len(wsi_paths))

        # ---- Build slide_id → predictions mapping -----------------------
        pred_map = self._load_predictions_dir(Path(predictions_dir))
        logger.info("Loaded predictions for %d slides.", len(pred_map))

        # ---- GT mapping (optional) --------------------------------------
        gt_map: dict[str, pd.DataFrame] = {}
        if gt_dir is not None:
            gt_map = self._load_predictions_dir(Path(gt_dir))

        output_dir_p = Path(output_dir)
        output_dir_p.mkdir(parents=True, exist_ok=True)

        # ---- Parallel execution -----------------------------------------
        results: dict[str, Path | Exception] = {}

        def _process(wsi_path: Path) -> tuple[str, Path | Exception]:
            sid = wsi_path.stem
            if sid not in pred_map:
                return sid, FileNotFoundError(f"No predictions found for '{sid}'")
            out_path = output_dir_p / f"{sid}_heatmap.png"
            gt_df    = gt_map.get(sid, None)
            try:
                saved = self.generate(
                    wsi_path        = wsi_path,
                    predictions_df  = pred_map[sid],
                    output_path     = out_path,
                    show_gt         = gt_df is not None,
                    gt_df           = gt_df,
                )
                return sid, saved
            except Exception as exc:
                return sid, exc

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_process, p): p.stem for p in wsi_paths}
            for fut in as_completed(futures):
                sid, outcome = fut.result()
                results[sid] = outcome
                if isinstance(outcome, Exception):
                    logger.error("Failed %s: %s", sid, outcome)
                else:
                    logger.info("  ✓ %s", outcome)

        n_ok  = sum(1 for v in results.values() if not isinstance(v, Exception))
        n_err = len(results) - n_ok
        logger.info(
            "Batch complete: %d generated, %d failed (no predictions: %d).",
            n_ok, n_err,
            sum(1 for v in results.values() if isinstance(v, FileNotFoundError)),
        )
        return results

    # ======================================================================
    # Private helpers
    # ======================================================================

    def _load_thumbnail(
        self, wsi_path: Path
    ) -> tuple[Image.Image, tuple[int, int]]:
        """Open WSI and return (RGB PIL thumbnail, (wsi_w, wsi_h))."""
        from ..datasets.wsi_utils import WSIReader
        with WSIReader(wsi_path) as reader:
            dims     = reader.get_dimensions()   # (w, h) at level 0
            thumbnail = reader.get_thumbnail(self.thumbnail_size)
        return thumbnail.convert("RGB"), dims

    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise column names; add ``patch_size`` and ``pred_class_int``."""
        df = df.copy()

        # ---- Predicted class → int --------------------------------------
        if "predicted_class" in df.columns:
            def _to_int(v):
                if isinstance(v, str):
                    return _NAME_TO_IDX.get(v.strip(), 0)
                return int(v)
            df["pred_class_int"] = df["predicted_class"].apply(_to_int)
        elif "pred_class" in df.columns:
            df["pred_class_int"] = pd.to_numeric(df["pred_class"], errors="coerce").fillna(0).astype(int)
        else:
            df["pred_class_int"] = 0

        # ---- Coordinates ------------------------------------------------
        df["x"] = pd.to_numeric(df["x"], errors="coerce").fillna(0).astype(int)
        df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0).astype(int)

        # ---- Patch size -------------------------------------------------
        if "patch_size" not in df.columns:
            df["patch_size"] = 512
        else:
            df["patch_size"] = pd.to_numeric(df["patch_size"], errors="coerce").fillna(512).astype(int)

        # ---- Probability columns (fill missing with 0) ------------------
        for c in range(self.n_classes):
            col = _PROB_COLS[c]
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        return df

    def _build_prob_maps(
        self,
        df: pd.DataFrame,
        thumb_w: int,
        thumb_h: int,
        scale_x: float,
        scale_y: float,
        patch_size: int,
    ) -> np.ndarray:
        """Build per-class probability maps and apply Gaussian smoothing.

        Returns
        -------
        prob_maps : (n_classes, thumb_h, thumb_w) float32
            Smoothed probability for each class at each thumbnail pixel.
            Pixels not covered by any patch have probability 0 for all classes.
        """
        n_classes = self.n_classes
        prob_maps = np.zeros((n_classes, thumb_h, thumb_w), dtype=np.float32)

        # Precompute thumbnail-space patch rectangle dimensions
        ps_x = max(1, int(patch_size * scale_x))
        ps_y = max(1, int(patch_size * scale_y))

        # Precompute all patch coordinates
        xs = (df["x"].values * scale_x).astype(np.int32)
        ys = (df["y"].values * scale_y).astype(np.int32)

        # Extract probability matrix (N, C)
        prob_arr = df[[_PROB_COLS[c] for c in range(n_classes)]].values.astype(np.float32)

        for i in range(len(df)):
            x0 = int(xs[i])
            y0 = int(ys[i])
            x1 = min(thumb_w, x0 + ps_x)
            y1 = min(thumb_h, y0 + ps_y)
            if x0 >= thumb_w or y0 >= thumb_h or x1 <= 0 or y1 <= 0:
                continue
            prob_maps[:, y0:y1, x0:x1] = prob_arr[i, :, np.newaxis, np.newaxis]

        # Per-class Gaussian smoothing
        if self.gaussian_sigma > 0:
            for c in range(n_classes):
                prob_maps[c] = gaussian_filter(prob_maps[c], sigma=self.gaussian_sigma)

        return prob_maps

    def _blend_overlay(
        self,
        thumbnail: np.ndarray,
        prob_maps: np.ndarray,
    ) -> np.ndarray:
        """Composite the predicted color overlay onto the thumbnail.

        Parameters
        ----------
        thumbnail:
            ``(H, W, 3)`` uint8 RGB array.
        prob_maps:
            ``(n_classes, H, W)`` float32 smoothed probability maps.

        Returns
        -------
        Blended ``(H, W, 3)`` uint8 RGB array.
        """
        thumb_f = thumbnail.astype(np.float32) / 255.0  # (H, W, 3)

        # Argmax class and confidence per pixel
        pred_map = prob_maps.argmax(axis=0)   # (H, W) int
        conf_map = prob_maps.max(axis=0)      # (H, W) float — acts as alpha

        # Build colour overlay (H, W, 3) float32
        color_layer = np.zeros_like(thumb_f)
        for cls_idx in range(self.n_classes):
            color_rgb = self.subtype_colors.get(cls_idx, (128, 128, 128))
            mask = (pred_map == cls_idx)
            color_layer[mask] = np.array(color_rgb, dtype=np.float32) / 255.0

        # Effective per-pixel alpha = global alpha × per-pixel confidence
        eff_alpha = (self.overlay_alpha * conf_map)[:, :, np.newaxis]  # (H, W, 1)

        blended = thumb_f * (1.0 - eff_alpha) + color_layer * eff_alpha
        blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
        return blended

    def _wsi_level_prediction(
        self,
        df: pd.DataFrame,
    ) -> tuple[int, float, np.ndarray]:
        """Compute WSI-level predicted class from mean patch probabilities.

        Returns
        -------
        (wsi_pred_class, wsi_confidence, mean_probs_array)
            where ``mean_probs_array`` has shape ``(n_classes,)``.
        """
        n = self.n_classes
        mean_probs = np.zeros(n, dtype=np.float64)
        for c in range(n):
            col = _PROB_COLS[c]
            if col in df.columns:
                mean_probs[c] = float(df[col].mean())

        wsi_pred = int(mean_probs.argmax())
        wsi_conf = float(mean_probs[wsi_pred])
        return wsi_pred, wsi_conf, mean_probs

    def _load_predictions_dir(
        self,
        pred_dir: Path,
    ) -> dict[str, pd.DataFrame]:
        """Scan *pred_dir* recursively for CSVs and build ``{slide_id: df}``.

        Handles two layouts:
        - CSVs with a ``slide_id`` column (trainer output) — each slide is
          extracted as a separate DataFrame from the multi-slide file.
        - Per-slide CSVs where the filename stem encodes the slide ID
          (optionally with ``_inst_pred`` / ``_predictions`` suffixes).
        """
        result: dict[str, pd.DataFrame] = {}
        STRIP_SUFFIXES = ("_inst_pred", "_inst_predictions", "_predictions", "_heatmap")

        for csv_path in sorted(pred_dir.rglob("*.csv")):
            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:
                logger.debug("Skipping unreadable CSV %s: %s", csv_path, exc)
                continue

            if "slide_id" in df.columns:
                # Multi-slide file: split by slide_id
                for sid, group in df.groupby("slide_id"):
                    sid = str(sid)
                    result[sid] = self._normalize_df(group.reset_index(drop=True))
            else:
                # Per-slide file: derive slide_id from filename
                stem = csv_path.stem
                for suf in STRIP_SUFFIXES:
                    if stem.endswith(suf):
                        stem = stem[: -len(suf)]
                        break
                result[stem] = self._normalize_df(df)

        return result


# ---------------------------------------------------------------------------
# Module-level convenience function (backwards-compatible API)
# ---------------------------------------------------------------------------

def generate_heatmap(
    wsi_path: str | Path,
    inst_csv: str | Path,
    output_path: str | Path,
    gt_csv: str | Path | None = None,
    thumbnail_max: int = 1024,
    overlay_alpha: float = 0.50,
    gaussian_radius: float = 5.0,
    patch_size: int = 512,
) -> Path:
    """Convenience wrapper around :class:`HeatmapGenerator` for a single WSI.

    Parameters
    ----------
    wsi_path:
        Path to the ``.tif`` WSI file.
    inst_csv:
        Per-patch prediction CSV (see :class:`HeatmapGenerator` for formats).
    output_path:
        Destination ``.png`` path.
    gt_csv:
        Optional ground-truth annotation CSV.
    thumbnail_max:
        Max dimension (width or height) of the WSI thumbnail in pixels.
    overlay_alpha:
        Maximum overlay opacity.
    gaussian_radius:
        Gaussian smoothing sigma in thumbnail pixels.
    patch_size:
        Patch size in level-0 pixels (used when not present in CSV).

    Returns
    -------
    Path to the saved PNG.
    """
    gen = HeatmapGenerator(
        thumbnail_size=(thumbnail_max, thumbnail_max),
        overlay_alpha=overlay_alpha,
        gaussian_sigma=gaussian_radius,
    )
    df = pd.read_csv(inst_csv)
    gt_df = pd.read_csv(gt_csv) if gt_csv and Path(gt_csv).exists() else None

    return gen.generate(
        wsi_path       = wsi_path,
        predictions_df = df,
        output_path    = output_path,
        show_gt        = gt_df is not None,
        gt_df          = gt_df,
    )
