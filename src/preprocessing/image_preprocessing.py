"""Phase 2 — deterministic image preprocessing for classical feature extraction.

Every image is loaded as RGB, resized to a fixed square resolution and
converted to normalised float arrays:

* ``rgb``       — ``(H, W, 3)`` float32 in ``[0, 1]``
* ``grayscale`` — ``(H, W)``   float32 in ``[0, 1]``, Rec. 709 luma
* ``hsv``       — ``(H, W, 3)`` float32 in ``[0, 1]``, channel order H, S, V

Preprocessing is purely deterministic: it learns nothing and uses no
randomness. The Phase 3 feature extractors consume these arrays directly, so
this module is the single source of preprocessing behaviour for the project.

The target size of 128x128 is a deliberate compromise: with the Phase 3 HOG
configuration (9 orientations, 8x8-pixel cells, 2x2-cell blocks) it produces
8,100-dimensional HOG vectors (~262 MB float32 for the full 8,091-image
project dataset), whereas 256x256 would produce 34,596 dimensions (~1.1 GB)
before PCA without adding useful information for handcrafted features.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv

#: Fixed working resolution ``(width, height)`` for every image.
TARGET_SIZE: tuple[int, int] = (128, 128)

#: Lanczos resampling keeps edges sharp for gradient-based features.
RESIZE_FILTER = Image.Resampling.LANCZOS


@dataclass(frozen=True)
class PreprocessedImage:
    """Normalised representations of one preprocessed image.

    Attributes:
        image_id: The dataset filename (e.g. ``1000268201_693b08cb0e.jpg``).
        rgb: ``(H, W, 3)`` float32 RGB values in ``[0, 1]``.
        grayscale: ``(H, W)`` float32 luma values in ``[0, 1]``.
        hsv: ``(H, W, 3)`` float32 HSV values in ``[0, 1]``.
    """

    image_id: str
    rgb: np.ndarray
    grayscale: np.ndarray
    hsv: np.ndarray


def load_rgb(path: str | Path) -> Image.Image:
    """Load an image from disk and return it as an RGB PIL image.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file exists but cannot be decoded as an image.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except Exception as error:
        raise ValueError(f"Cannot decode image '{path}': {error}") from error


def resize_image(
    image: Image.Image, size: tuple[int, int] = TARGET_SIZE
) -> Image.Image:
    """Resize ``image`` to ``size`` (width, height) with Lanczos resampling.

    Raises:
        ValueError: If ``size`` is not strictly positive.
    """
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError(f"Target size must be positive, got {size}")
    return image.resize(size, resample=RESIZE_FILTER)


def image_to_array(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to a float32 array in ``[0, 1]``, shape ``(H, W, C)``."""
    return np.asarray(image, dtype=np.float32) / 255.0


def _require_rgb(rgb: np.ndarray) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected an RGB array of shape (H, W, 3), got {rgb.shape}")
    if rgb.size == 0:
        raise ValueError("Cannot convert an empty image")


def to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB array in ``[0, 1]`` to luma grayscale in ``[0, 1]``.

    Uses the Rec. 709 luma coefficients implemented by
    ``skimage.color.rgb2gray``: ``Y = 0.2125 R + 0.7154 G + 0.0721 B``.
    """
    _require_rgb(rgb)
    return rgb2gray(rgb).astype(np.float32)


def to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB array in ``[0, 1]`` to HSV in ``[0, 1]`` (H, S, V order)."""
    _require_rgb(rgb)
    return rgb2hsv(rgb).astype(np.float32)


def preprocess_image(
    path: str | Path, image_id: str | None = None
) -> PreprocessedImage:
    """Load, resize and convert one image.

    This is the single entry point consumed by the Phase 3 feature
    extractors, guaranteeing that every image receives identical
    preprocessing.

    Args:
        path: Path to the image file.
        image_id: Dataset filename to attach; defaults to ``path.name``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be decoded.
    """
    path = Path(path)
    resized = resize_image(load_rgb(path))
    rgb = image_to_array(resized)
    return PreprocessedImage(
        image_id=image_id if image_id is not None else path.name,
        rgb=rgb,
        grayscale=to_grayscale(rgb),
        hsv=to_hsv(rgb),
    )
