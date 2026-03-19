"""
WSI reading utilities with .tif support.

Provides a unified interface for opening whole-slide images that:
  1. Tries OpenSlide first (pyramidal TIFF, BigTIFF, SVS, etc.)
  2. Falls back to tifffile for non-pyramidal multi-page TIFF
  3. Falls back to PIL for standard flat TIFF / PNG

Critical: all file patterns use *.tif — NOT *.svs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WSI reader wrapper
# ---------------------------------------------------------------------------

class WSIReader:
    """Unified WSI reader that wraps OpenSlide / tifffile / PIL.

    Usage::

        reader = WSIReader(path)
        thumbnail = reader.get_thumbnail(max_size=1024)
        region = reader.read_region(x=0, y=0, level=0, width=512, height=512)
        reader.close()

    Or as a context manager::

        with WSIReader(path) as reader:
            ...
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._slide: Any = None
        self._backend: str = ""
        self._open()

    # ------------------------------------------------------------------
    # Internal: open with best available backend
    # ------------------------------------------------------------------
    def _open(self) -> None:
        # 1. Try OpenSlide (handles pyramidal TIFF / SVS / NDPI etc.)
        try:
            import openslide

            self._slide = openslide.OpenSlide(str(self.path))
            self._backend = "openslide"
            return
        except Exception as exc:
            logger.debug("OpenSlide failed for %s: %s", self.path.name, exc)

        # 2. Try tifffile (multi-page or non-pyramidal TIFF)
        try:
            import tifffile

            self._slide = tifffile.TiffFile(str(self.path))
            self._backend = "tifffile"
            return
        except Exception as exc:
            logger.debug("tifffile failed for %s: %s", self.path.name, exc)

        # 3. Fall back to PIL
        try:
            self._slide = Image.open(str(self.path))
            self._backend = "pil"
            return
        except Exception as exc:
            raise OSError(
                f"Cannot open {self.path} with any supported backend "
                f"(OpenSlide, tifffile, PIL): {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    @property
    def backend(self) -> str:
        return self._backend

    @property
    def dimensions(self) -> tuple[int, int]:
        """Returns (width, height) at level 0."""
        if self._backend == "openslide":
            return self._slide.dimensions  # (w, h)
        if self._backend == "tifffile":
            page = self._slide.pages[0]
            return (page.imagewidth, page.imagelength)
        if self._backend == "pil":
            return self._slide.size  # (w, h)
        raise RuntimeError("No backend loaded.")

    def get_thumbnail(self, max_size: int = 1024) -> np.ndarray:
        """Return an RGB thumbnail as a numpy array (H, W, 3)."""
        w, h = self.dimensions
        scale = max_size / max(w, h)
        thumb_w = max(1, int(w * scale))
        thumb_h = max(1, int(h * scale))

        if self._backend == "openslide":
            thumb = self._slide.get_thumbnail((thumb_w, thumb_h))
            return np.array(thumb.convert("RGB"))

        if self._backend == "tifffile":
            page = self._slide.pages[0]
            img_arr = page.asarray()
            img = Image.fromarray(img_arr)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = img.resize((thumb_w, thumb_h), Image.LANCZOS)
            return np.array(img)

        if self._backend == "pil":
            img = self._slide.copy().convert("RGB")
            img = img.resize((thumb_w, thumb_h), Image.LANCZOS)
            return np.array(img)

        raise RuntimeError("No backend loaded.")

    def read_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        level: int = 0,
    ) -> np.ndarray:
        """Read a rectangular region and return as RGB numpy array (H, W, 3).

        Args:
            x, y:   Top-left corner at level 0 coordinates.
            width, height: Region size in pixels at level 0.
            level:  Pyramid level (OpenSlide only; ignored for other backends).
        """
        if self._backend == "openslide":
            region = self._slide.read_region((x, y), level, (width, height))
            return np.array(region.convert("RGB"))

        if self._backend in ("tifffile", "pil"):
            # For flat TIFF / PIL: load full image and crop (only feasible for
            # small WSIs; large flat TIFFs should be converted to pyramidal).
            w_full, h_full = self.dimensions
            if self._backend == "tifffile":
                img_arr = self._slide.pages[0].asarray()
                img = Image.fromarray(img_arr).convert("RGB")
            else:
                img = self._slide.convert("RGB")
            region = img.crop((x, y, min(x + width, w_full), min(y + height, h_full)))
            region = region.resize((width, height), Image.BILINEAR)
            return np.array(region)

        raise RuntimeError("No backend loaded.")

    def close(self) -> None:
        if self._backend == "openslide" and self._slide is not None:
            self._slide.close()
        elif self._backend == "tifffile" and self._slide is not None:
            self._slide.close()
        self._slide = None

    def __enter__(self) -> "WSIReader":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Convenience functions (module-level API)
# ---------------------------------------------------------------------------

def open_wsi(path: str | Path) -> WSIReader:
    """Open a WSI file and return a WSIReader instance."""
    return WSIReader(path)


def get_thumbnail(path: str | Path, max_size: int = 1024) -> np.ndarray:
    """Shortcut: open a WSI and return its thumbnail as an RGB numpy array."""
    with WSIReader(path) as reader:
        return reader.get_thumbnail(max_size=max_size)
