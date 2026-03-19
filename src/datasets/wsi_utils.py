"""
WSI reading utilities with .tif support.

Provides a unified interface for opening whole-slide images that:
  1. Tries OpenSlide first (pyramidal/tiled TIFF, BigTIFF, SVS, etc.)
  2. Falls back to tifffile for non-pyramidal multi-page TIFF
  3. Falls back to PIL for standard flat TIFF

Critical: all file patterns use *.tif — NOT *.svs.

Public API
----------
WSIReader          — context-manager-compatible WSI reader class
is_tissue()        — Otsu-based tissue vs. background classifier
tessellate_wsi()   — enumerate foreground patch coordinates for a WSI
open_wsi()         — convenience constructor
get_thumbnail()    — one-liner thumbnail helper
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend detection helpers
# ---------------------------------------------------------------------------

def _try_openslide(path: Path):
    """Return an openslide.OpenSlide object or raise."""
    import openslide  # ImportError propagates if not installed
    slide = openslide.OpenSlide(str(path))
    # Verify that at least one level is readable — openslide can open
    # some non-pyramidal TIFFs without raising but silently produce
    # garbage dimensions; a quick sanity check guards against that.
    w, h = slide.dimensions
    if w <= 0 or h <= 0:
        slide.close()
        raise ValueError(f"OpenSlide returned invalid dimensions ({w}×{h})")
    return slide


def _try_tifffile(path: Path):
    """Return a tifffile.TiffFile object or raise."""
    import tifffile
    tf = tifffile.TiffFile(str(path))
    if not tf.pages:
        tf.close()
        raise ValueError("tifffile: no pages found")
    return tf


def _tifffile_full_image(tf) -> Image.Image:
    """Load the largest (first) page from a TiffFile as an RGB PIL Image."""
    arr = tf.pages[0].asarray()
    img = Image.fromarray(arr)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


# ---------------------------------------------------------------------------
# WSIReader
# ---------------------------------------------------------------------------

class WSIReader:
    """Unified WSI reader that wraps OpenSlide / tifffile / PIL.

    Backend selection order
    -----------------------
    1. openslide — fastest; supports pyramidal TIFF, BigTIFF, SVS, NDPI …
    2. tifffile  — for non-pyramidal / stripped TIFF files
    3. PIL       — last-resort fallback for any image PIL can open

    For tifffile and PIL backends the full image is loaded into memory on
    the first ``read_region`` call and cached; subsequent calls slice the
    cached array.  This is acceptable for the relatively small flat-TIFF
    files encountered in practice; very large flat TIFFs should be
    converted to pyramidal TIFF before use.

    Parameters
    ----------
    wsi_path:
        Path to the .tif file.

    Examples
    --------
    >>> reader = WSIReader("slide.tif")
    >>> w, h = reader.get_dimensions()
    >>> thumb = reader.get_thumbnail((512, 512))
    >>> region = reader.read_region((0, 0), level=0, size=(512, 512))
    >>> reader.close()

    Or as a context manager:

    >>> with WSIReader("slide.tif") as r:
    ...     thumb = r.get_thumbnail((512, 512))
    """

    def __init__(self, wsi_path: str | Path) -> None:
        self.path = Path(wsi_path)
        self._slide: Any = None
        self._backend: str = ""
        # Lazy cache for non-openslide backends (PIL Image, kept open)
        self._pil_cache: Image.Image | None = None
        self._open()

    # ------------------------------------------------------------------
    # Backend initialisation
    # ------------------------------------------------------------------

    def _open(self) -> None:
        # 1. OpenSlide
        try:
            self._slide = _try_openslide(self.path)
            self._backend = "openslide"
            logger.debug("Opened %s via OpenSlide.", self.path.name)
            return
        except Exception as exc:
            logger.debug("OpenSlide failed for %s: %s", self.path.name, exc)

        # 2. tifffile
        try:
            self._slide = _try_tifffile(self.path)
            self._backend = "tifffile"
            logger.debug("Opened %s via tifffile.", self.path.name)
            return
        except Exception as exc:
            logger.debug("tifffile failed for %s: %s", self.path.name, exc)

        # 3. PIL
        try:
            self._slide = Image.open(str(self.path))
            self._backend = "pil"
            logger.debug("Opened %s via PIL.", self.path.name)
            return
        except Exception as exc:
            raise OSError(
                f"Cannot open {self.path} with any supported backend "
                f"(OpenSlide, tifffile, PIL). Last error: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def backend(self) -> str:
        """Name of the active backend: 'openslide', 'tifffile', or 'pil'."""
        return self._backend

    def get_dimensions(self) -> tuple[int, int]:
        """Return (width, height) at level 0 in pixels."""
        if self._backend == "openslide":
            return self._slide.dimensions  # already (w, h)
        if self._backend == "tifffile":
            page = self._slide.pages[0]
            return (int(page.imagewidth), int(page.imagelength))
        if self._backend == "pil":
            return self._slide.size  # (w, h)
        raise RuntimeError("No backend loaded.")

    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> Image.Image:
        """Read a rectangular region and return as an RGB PIL Image.

        Parameters
        ----------
        location:
            (x, y) top-left corner in level-0 pixel coordinates.
        level:
            Pyramid level (only meaningful for the openslide backend;
            ignored by tifffile / PIL which always operate at level 0).
        size:
            (width, height) of the region to read, in pixels at the
            requested *level* (not level 0).

        Returns
        -------
        PIL.Image (mode RGB)
        """
        x, y = location
        w, h = size

        if self._backend == "openslide":
            region = self._slide.read_region((x, y), level, (w, h))
            return region.convert("RGB")

        # tifffile / PIL: operate on the cached full image
        full = self._get_pil_cache()
        fw, fh = full.size
        x1 = min(x + w, fw)
        y1 = min(y + h, fh)
        region = full.crop((x, y, x1, y1))
        if region.size != (w, h):
            region = region.resize((w, h), Image.BILINEAR)
        return region.convert("RGB")

    def get_thumbnail(self, size: tuple[int, int]) -> Image.Image:
        """Return a downscaled RGB thumbnail of (at most) the given size.

        The aspect ratio is preserved; the result fits within *size* but
        may be smaller in one dimension.

        Parameters
        ----------
        size:
            (max_width, max_height) bounding box for the thumbnail.

        Returns
        -------
        PIL.Image (mode RGB)
        """
        tw, th = size
        if self._backend == "openslide":
            thumb = self._slide.get_thumbnail((tw, th))
            return thumb.convert("RGB")

        full = self._get_pil_cache()
        full.thumbnail((tw, th), Image.LANCZOS)
        # thumbnail() modifies in place for non-lazy images; return a copy
        # so the cache is not mutated.
        return full.copy().convert("RGB")

    def get_downscale_factor(self, thumbnail_size: tuple[int, int]) -> float:
        """Return the downscale factor applied when generating a thumbnail.

        This is the ratio of the thumbnail's *longer* edge to the WSI's
        corresponding dimension at level 0.  Multiply a thumbnail pixel
        coordinate by ``1 / downscale_factor`` to recover level-0 coords.

        Parameters
        ----------
        thumbnail_size:
            The (max_width, max_height) passed to ``get_thumbnail``.

        Returns
        -------
        float in (0, 1]
        """
        wsi_w, wsi_h = self.get_dimensions()
        tw, th = thumbnail_size
        return min(tw / wsi_w, th / wsi_h, 1.0)

    def close(self) -> None:
        """Release all held file handles and clear caches."""
        if self._backend == "openslide" and self._slide is not None:
            self._slide.close()
        elif self._backend == "tifffile" and self._slide is not None:
            self._slide.close()
        elif self._backend == "pil" and self._slide is not None:
            self._slide.close()
        self._slide = None
        self._pil_cache = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "WSIReader":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"WSIReader(path={self.path.name!r}, backend={self._backend!r}, "
            f"dimensions={self.get_dimensions()})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_pil_cache(self) -> Image.Image:
        """Return (and lazily load) the full-image PIL cache.

        Only used for tifffile / PIL backends.
        """
        if self._pil_cache is not None:
            return self._pil_cache

        if self._backend == "tifffile":
            self._pil_cache = _tifffile_full_image(self._slide)
        elif self._backend == "pil":
            # Force full decode and detach from the file handle so the
            # original Image object can be used independently.
            self._pil_cache = self._slide.convert("RGB")
        else:
            raise RuntimeError(
                "_get_pil_cache called on an openslide-backed reader."
            )
        return self._pil_cache


# ---------------------------------------------------------------------------
# Tissue detection
# ---------------------------------------------------------------------------

def is_tissue(
    patch: Image.Image | np.ndarray,
    threshold: float = 0.7,
) -> bool:
    """Return True if *patch* contains tissue (i.e. is not background).

    Method
    ------
    1. Convert patch to grayscale.
    2. Apply Otsu thresholding to separate foreground (tissue, dark) from
       background (slide glass, white/near-white).
    3. The patch is considered tissue when the fraction of *foreground*
       pixels exceeds *threshold*.

    Parameters
    ----------
    patch:
        RGB image as a PIL Image or a numpy array of shape (H, W, 3),
        dtype uint8.
    threshold:
        Minimum foreground pixel fraction to be classified as tissue.
        0.7 means at least 70 % of pixels must be non-background.

    Returns
    -------
    bool
    """
    if isinstance(patch, Image.Image):
        arr = np.array(patch.convert("RGB"))
    else:
        arr = patch

    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # binary == 255 → foreground (tissue); == 0 → background (white)
    foreground_frac = float(binary.mean()) / 255.0
    return foreground_frac >= threshold


# ---------------------------------------------------------------------------
# WSI tessellation
# ---------------------------------------------------------------------------

def tessellate_wsi(
    wsi_path: str | Path,
    patch_size: int = 512,
    step_size: int = 512,
    level: int = 0,
    tissue_threshold: float = 0.7,
) -> list[dict]:
    """Enumerate tissue patch coordinates for a whole-slide image.

    Scans the WSI in a regular grid (stride = *step_size*), reads each
    candidate patch, and keeps those that pass the tissue test.

    Parameters
    ----------
    wsi_path:
        Path to the .tif WSI file.
    patch_size:
        Patch size in pixels *at level 0*.
    step_size:
        Grid stride in pixels *at level 0*.  Use ``step_size == patch_size``
        for non-overlapping patches (default).
    level:
        Pyramid level at which patches are read (passed to
        ``WSIReader.read_region``).  Coordinates are always in level-0
        pixel space.
    tissue_threshold:
        Forwarded to ``is_tissue``; fraction of foreground pixels required.

    Returns
    -------
    list of dicts, each with keys:
        ``x``          — left edge of the patch in level-0 pixels
        ``y``          — top edge of the patch in level-0 pixels
        ``patch_size`` — patch size in pixels (always equals *patch_size*)

    Notes
    -----
    * Patches that extend beyond the WSI boundary are skipped (no padding).
    * The list is ordered row-first (y outer, x inner).
    """
    wsi_path = Path(wsi_path)
    patches: list[dict] = []

    with WSIReader(wsi_path) as reader:
        wsi_w, wsi_h = reader.get_dimensions()

        xs = range(0, wsi_w - patch_size + 1, step_size)
        ys = range(0, wsi_h - patch_size + 1, step_size)

        for y in ys:
            for x in xs:
                patch_img = reader.read_region(
                    location=(x, y),
                    level=level,
                    size=(patch_size, patch_size),
                )
                if is_tissue(patch_img, threshold=tissue_threshold):
                    patches.append({"x": x, "y": y, "patch_size": patch_size})

    logger.info(
        "%s: %d tissue patches out of %d candidates (patch=%d, step=%d).",
        wsi_path.name,
        len(patches),
        len(list(range(0, wsi_w - patch_size + 1, step_size)))
        * len(list(range(0, wsi_h - patch_size + 1, step_size))),
        patch_size,
        step_size,
    )
    return patches


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def open_wsi(path: str | Path) -> WSIReader:
    """Open a WSI file and return a ``WSIReader`` instance."""
    return WSIReader(path)


def get_thumbnail(
    path: str | Path,
    size: tuple[int, int] = (1024, 1024),
) -> Image.Image:
    """Open a WSI and return its thumbnail as a PIL Image.

    Parameters
    ----------
    path:
        Path to the .tif WSI file.
    size:
        ``(max_width, max_height)`` bounding box for the thumbnail.
    """
    with WSIReader(path) as reader:
        return reader.get_thumbnail(size)
