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
        wsi_path: str | Path | None,
        predictions_df: pd.DataFrame,
        output_path: str | Path,
        show_gt: bool = False,
        gt_df: pd.DataFrame | None = None,
    ) -> Path:
        """Generate and save a heatmap PNG for one WSI.

        Parameters
        ----------
        wsi_path:
            Path to the ``.tif`` WSI file, or ``None`` / a non-existent path
            to run in *no-WSI mode*: a white canvas sized to the patch
            coordinate bounding box is used instead of a tissue thumbnail.
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

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        wsi_path = Path(wsi_path) if wsi_path is not None else None
        no_wsi   = wsi_path is None or not wsi_path.exists()
        slide_id = wsi_path.stem if wsi_path is not None else Path(str(output_path)).stem.replace("_heatmap", "")

        # ---- Normalise DataFrames ----------------------------------------
        preds = self._normalize_df(predictions_df)

        # ---- Load WSI thumbnail (or synthetic canvas) --------------------
        if no_wsi:
            thumbnail_pil, (wsi_w, wsi_h) = self._synthetic_canvas(preds)
            if wsi_path is not None:
                logger.warning(
                    "WSI file not found for '%s' — using synthetic canvas. "
                    "Heatmap shows spatial layout of patches only (no tissue background).",
                    slide_id,
                )
        else:
            thumbnail_pil, (wsi_w, wsi_h) = self._load_thumbnail(wsi_path)

        thumb_arr = np.array(thumbnail_pil)          # (H, W, 3) uint8
        thumb_h, thumb_w = thumb_arr.shape[:2]

        scale_x = thumb_w / wsi_w
        scale_y = thumb_h / wsi_h

        # ---- Infer patch size (majority value, fallback 512) -------------
        patch_size = int(preds["patch_size"].mode().iloc[0]) if "patch_size" in preds.columns else 512

        # ---- Build the overlay. Default: blocky subtype patch mask (matches
        #      SMMILe paper Fig. 5 — discrete tumor patches). Fallbacks:
        #      probability → attention → dense class map, used only when the
        #      segmentation columns are missing. ------------------------------
        subtype_class = None
        seg_area = None
        seg_result = self._build_segmentation_overlay(
            preds, thumb_arr, scale_x, scale_y, patch_size
        )
        if seg_result is not None:
            overlay_rgb, seg_area = seg_result
            mode = "seg"
        elif (prob_result := self._build_subtype_prob_overlay(
                preds, thumb_arr, scale_x, scale_y, patch_size)) is not None:
            overlay_rgb, subtype_class = prob_result
            mode = "prob"
        elif (attn_result := self._build_attention_overlay(
                preds, thumb_arr, scale_x, scale_y, patch_size)) is not None:
            overlay_rgb, subtype_class = attn_result
            mode = "attn"
        else:
            prob_maps = self._build_prob_maps(preds, thumb_w, thumb_h, scale_x, scale_y, patch_size)
            overlay_rgb = self._blend_overlay(thumb_arr, prob_maps)
            mode = "class"

        # ---- Save the overlay image directly (single image, no side-by-side) -
        Image.fromarray(overlay_rgb).save(str(output_path))
        logger.info("Saved heatmap → %s", output_path)
        return output_path

    def generate_batch(
        self,
        wsi_dir: str | Path | None,
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

        # ---- Build slide_id → predictions mapping -----------------------
        pred_map = self._load_predictions_dir(Path(predictions_dir))
        logger.info("Loaded predictions for %d slides.", len(pred_map))

        # ---- Resolve WSI paths (optional) --------------------------------
        wsi_map: dict[str, Path] = {}
        if wsi_dir is not None:
            wsi_dir_s = str(wsi_dir)
            if "*" in wsi_dir_s or "?" in wsi_dir_s:
                wsi_paths = sorted(Path(p) for p in glob.glob(wsi_dir_s))
            else:
                wsi_paths = sorted(Path(wsi_dir_s).glob("*.tif"))
            wsi_map = {p.stem: p for p in wsi_paths}
            if wsi_map:
                logger.info("Found %d WSI file(s).", len(wsi_map))
            else:
                logger.warning("No .tif files found in: %s — running in no-WSI mode.", wsi_dir_s)
        else:
            logger.info("No wsi_dir provided — running in no-WSI mode (synthetic canvas).")

        # ---- GT mapping (optional) --------------------------------------
        gt_map: dict[str, pd.DataFrame] = {}
        if gt_dir is not None:
            gt_map = self._load_predictions_dir(Path(gt_dir))

        output_dir_p = Path(output_dir)
        output_dir_p.mkdir(parents=True, exist_ok=True)

        # ---- Parallel execution (iterate over prediction slide IDs) ------
        results: dict[str, Path | Exception] = {}

        def _process(sid: str) -> tuple[str, Path | Exception]:
            wsi_path = wsi_map.get(sid)   # None → no-WSI mode for this slide
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
            futures = {pool.submit(_process, sid): sid for sid in pred_map}
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

    def _synthetic_canvas(
        self, df: pd.DataFrame
    ) -> tuple[Image.Image, tuple[int, int]]:
        """Create a white canvas sized to the patch coordinate bounding box.

        Used when no WSI file is available.  The canvas preserves the spatial
        layout of patches so heatmap colors are positioned correctly relative
        to each other, even without the tissue background.

        Returns the same ``(PIL.Image, (wsi_w, wsi_h))`` tuple as
        ``_load_thumbnail`` so the rest of ``generate()`` is unchanged.
        """
        patch_size = int(df["patch_size"].mode().iloc[0]) if "patch_size" in df.columns else 512
        xs = pd.to_numeric(df["x"], errors="coerce").fillna(0).astype(int)
        ys = pd.to_numeric(df["y"], errors="coerce").fillna(0).astype(int)

        wsi_w = int(xs.max()) + patch_size
        wsi_h = int(ys.max()) + patch_size

        # Scale to thumbnail_size while preserving aspect ratio
        max_dim = max(self.thumbnail_size)
        scale   = min(max_dim / wsi_w, max_dim / wsi_h)
        thumb_w = max(1, int(wsi_w * scale))
        thumb_h = max(1, int(wsi_h * scale))

        # Light-gray canvas (distinguishable from white background regions)
        canvas = Image.new("RGB", (thumb_w, thumb_h), color=(220, 220, 220))
        return canvas, (wsi_w, wsi_h)

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

    def _build_subtype_prob_overlay(
        self,
        preds: pd.DataFrame,
        thumbnail: np.ndarray,
        scale_x: float,
        scale_y: float,
        patch_size: int,
    ):
        """Single-subtype spatial-probability heatmap (consistent across slides).

        For a slide predicted (at WSI level) as subtype *S*, colors the tissue by
        the per-patch probability of *S* using a fixed ``jet`` scale — so every
        heatmap reads the same way (red = strong evidence for the subtype named
        in the title), regardless of which subtype it is. Background/normal
        patches (``is_background == 1``) are not colored. Returns
        ``(overlay_rgb, class_name)`` or ``None`` when the needed columns are
        absent.
        """
        if "bag_pred_class" not in preds.columns or len(preds) == 0:
            return None
        cls_name = str(preds["bag_pred_class"].iloc[0])
        prob_col = f"prob_{cls_name}"
        if prob_col not in preds.columns:
            return None

        import matplotlib.cm as cm

        thumb_h, thumb_w = thumbnail.shape[:2]
        ps_x = max(1, int(patch_size * scale_x))
        ps_y = max(1, int(patch_size * scale_y))
        xs = (preds["x"].values * scale_x).astype(np.int32)
        ys = (preds["y"].values * scale_y).astype(np.int32)
        vals = pd.to_numeric(preds[prob_col], errors="coerce").fillna(0.0).values.astype(np.float32)
        # Restrict the map to foreground (tumor) patches so normal tissue and the
        # background dilution do not wash out the signal.
        if "is_background" in preds.columns:
            fg = pd.to_numeric(preds["is_background"], errors="coerce").fillna(0).values.astype(int) == 0
        else:
            fg = np.ones(len(preds), dtype=bool)

        acc = np.zeros((thumb_h, thumb_w), dtype=np.float32)
        cnt = np.zeros((thumb_h, thumb_w), dtype=np.float32)
        for i in range(len(preds)):
            if not fg[i]:
                continue
            x0, y0 = int(xs[i]), int(ys[i])
            x1, y1 = min(thumb_w, x0 + ps_x), min(thumb_h, y0 + ps_y)
            if x0 >= thumb_w or y0 >= thumb_h or x1 <= 0 or y1 <= 0:
                continue
            acc[y0:y1, x0:x1] += vals[i]
            cnt[y0:y1, x0:x1] += 1.0
        fgmask = cnt > 0
        if not fgmask.any():
            return None

        # Local mean prob among foreground patches (coverage = foreground density).
        sigma = max(self.gaussian_sigma, ps_x * 1.2, ps_y * 1.2)
        with np.errstate(invalid="ignore", divide="ignore"):
            val0 = np.where(fgmask, acc / np.maximum(cnt, 1.0), 0.0)
            val_s = gaussian_filter(val0, sigma=sigma)
            cov_s = gaussian_filter(fgmask.astype(np.float32), sigma=sigma)
            field = np.where(cov_s > 1e-3, val_s / np.maximum(cov_s, 1e-3), 0.0)

        # Fixed absolute scale so a given color means the same probability on
        # every slide (P=0.5 → full red).
        norm = np.clip(field / 0.5, 0.0, 1.0)
        fg_w = np.clip((cov_s - 0.2) / 0.3, 0.0, 1.0)        # confine to foreground

        colored = cm.get_cmap("jet")(norm)[:, :, :3].astype(np.float32)
        thumb_f = thumbnail.astype(np.float32) / 255.0
        # Alpha gates on probability (low → transparent, so no blue haze) AND on
        # foreground coverage.
        alpha = ((norm ** 1.8) * fg_w * 0.95)[:, :, None]
        blended = thumb_f * (1.0 - alpha) + colored * alpha
        overlay_rgb = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
        return overlay_rgb, cls_name

    def _build_segmentation_overlay(
        self,
        preds: pd.DataFrame,
        thumbnail: np.ndarray,
        scale_x: float,
        scale_y: float,
        patch_size: int,
    ):
        """Sparse instance-segmentation overlay (matches SMMILe paper Fig. 5).

        SMMILe's refinement classifies each patch into a subtype **or a
        background/negative class**. Only foreground (``is_background == 0``)
        patches are colored, by predicted subtype; background patches stay as
        tissue. A heavy Gaussian blur merges the patch squares into coherent,
        morphology-scale regions rather than a checkerboard. Returns
        ``(overlay_rgb, area_fractions)`` or ``None`` when ``is_background`` is
        absent (older CSVs) or no foreground patches exist.
        """
        if "is_background" not in preds.columns:
            return None

        thumb_h, thumb_w = thumbnail.shape[:2]
        ps_x = max(1, int(patch_size * scale_x))
        ps_y = max(1, int(patch_size * scale_y))
        xs = (preds["x"].values * scale_x).astype(np.int32)
        ys = (preds["y"].values * scale_y).astype(np.int32)
        is_bg = pd.to_numeric(preds["is_background"], errors="coerce").fillna(0).values.astype(int)
        cls   = preds["pred_class_int"].values.astype(int)

        # Color every foreground (tumor) patch with the WSI-predicted subtype's
        # single color — one color per slide, so the tumor-area mask is clean
        # and consistent instead of a per-patch rainbow.
        bag_idx = None
        if "bag_pred_class" in preds.columns and len(preds):
            bag_idx = _NAME_TO_IDX.get(str(preds["bag_pred_class"].iloc[0]))
        color_map = np.zeros((thumb_h, thumb_w, 3), dtype=np.float32)
        alpha_map = np.zeros((thumb_h, thumb_w), dtype=np.float32)
        counts = np.zeros(self.n_classes, dtype=np.int64)
        for i in range(len(preds)):
            if is_bg[i]:
                continue
            x0, y0 = int(xs[i]), int(ys[i])
            x1, y1 = min(thumb_w, x0 + ps_x), min(thumb_h, y0 + ps_y)
            if x0 >= thumb_w or y0 >= thumb_h or x1 <= 0 or y1 <= 0:
                continue
            c = bag_idx if bag_idx is not None else int(cls[i])
            color = np.array(self.subtype_colors.get(c, (128, 128, 128)), dtype=np.float32) / 255.0
            color_map[y0:y1, x0:x1] = color
            alpha_map[y0:y1, x0:x1] = 1.0
            if 0 <= c < self.n_classes:
                counts[c] += 1
        if counts.sum() == 0:
            return None

        # Blocky patch mask (matches SMMILe paper Fig. 5): keep the discrete
        # patch squares — NO region-merging blur. ``color_map`` already holds
        # each foreground patch's subtype color; ``alpha_map`` marks painted
        # (tumor) pixels. Semi-transparent so the H&E tissue shows through.
        a = (np.clip(alpha_map, 0.0, 1.0) * 0.5)[:, :, None]
        thumb_f = thumbnail.astype(np.float32) / 255.0
        blended = thumb_f * (1.0 - a) + color_map * a
        overlay_rgb = np.clip(blended * 255.0, 0, 255).astype(np.uint8)

        area = counts / counts.sum()
        return overlay_rgb, area

    def _build_attention_overlay(
        self,
        preds: pd.DataFrame,
        thumbnail: np.ndarray,
        scale_x: float,
        scale_y: float,
        patch_size: int,
    ):
        """Build a smooth attention heatmap overlay for the WSI-predicted class.

        Uses the per-instance attention weights (``attn_<CLASS>`` columns saved by
        the trainer). Attention is highly concentrated on discriminative patches,
        so this shows *where the model looks* — unlike per-patch class confidence,
        which is near-uniform. Returns ``(overlay_rgb, class_name)`` or ``None``
        when attention columns are absent.
        """
        # WSI-level predicted subtype drives which attention map to show.
        if "bag_pred_class" not in preds.columns or len(preds) == 0:
            return None
        cls_name = str(preds["bag_pred_class"].iloc[0])
        attn_col = f"attn_{cls_name}"
        if attn_col not in preds.columns:
            return None

        import matplotlib.cm as cm

        thumb_h, thumb_w = thumbnail.shape[:2]
        ps_x = max(1, int(patch_size * scale_x))
        ps_y = max(1, int(patch_size * scale_y))
        xs = (preds["x"].values * scale_x).astype(np.int32)
        ys = (preds["y"].values * scale_y).astype(np.int32)
        vals = pd.to_numeric(preds[attn_col], errors="coerce").fillna(0.0).values.astype(np.float32)

        # Paint each patch's attention into its thumbnail rectangle (mean over overlaps).
        acc = np.zeros((thumb_h, thumb_w), dtype=np.float32)
        cnt = np.zeros((thumb_h, thumb_w), dtype=np.float32)
        for i in range(len(preds)):
            x0, y0 = int(xs[i]), int(ys[i])
            x1, y1 = min(thumb_w, x0 + ps_x), min(thumb_h, y0 + ps_y)
            if x0 >= thumb_w or y0 >= thumb_h or x1 <= 0 or y1 <= 0:
                continue
            acc[y0:y1, x0:x1] += vals[i]
            cnt[y0:y1, x0:x1] += 1.0
        tissue = cnt > 0

        # Normalised-convolution smoothing: blur value and coverage, then divide,
        # so patch blocks merge into coherent regions without edge dimming.
        sigma = max(self.gaussian_sigma, ps_x * 1.2, ps_y * 1.2)
        with np.errstate(invalid="ignore", divide="ignore"):
            val0 = np.where(tissue, acc / np.maximum(cnt, 1.0), 0.0)
            val_s = gaussian_filter(val0, sigma=sigma)
            cov_s = gaussian_filter(tissue.astype(np.float32), sigma=sigma)
            field = np.where(cov_s > 1e-3, val_s / np.maximum(cov_s, 1e-3), 0.0)

        # Normalise to [0, 1] by a high percentile of tissue values (robust to
        # a few extreme patches), then gamma-boost mid-tones for visibility.
        tv = field[tissue]
        hi = float(np.percentile(tv, 99.0)) if tv.size else 0.0
        if hi <= 0:
            hi = float(field.max()) or 1.0
        norm = np.clip(field / (hi + 1e-8), 0.0, 1.0) ** 0.6
        # Confine the overlay to tissue: a soft coverage ramp prevents the
        # normalised-convolution division from blooming hot blobs past the
        # tissue edge into the background.
        tissue_w = np.clip((cov_s - 0.2) / 0.3, 0.0, 1.0)
        norm = norm * tissue_w

        # Composite jet colormap onto the tissue; alpha = attention (cold → tissue).
        colored = cm.get_cmap("jet")(norm)[:, :, :3].astype(np.float32)
        thumb_f = thumbnail.astype(np.float32) / 255.0
        alpha = (norm * 0.85)[:, :, None]
        blended = thumb_f * (1.0 - alpha) + colored * alpha
        overlay_rgb = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
        return overlay_rgb, cls_name

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
