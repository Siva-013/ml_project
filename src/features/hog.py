"""Phase 3 — Histogram of Oriented Gradients (classical, deterministic).

HOG summarises local shape and edge structure:

1. Compute per-pixel gradients (magnitude and orientation).
2. Vote the gradient magnitudes into orientation histograms over small
   spatial *cells* (8x8 pixels, 9 orientation bins covering 0-180 degrees).
3. Normalise histograms over overlapping *blocks* of 2x2 cells with the
   L2-Hys scheme (L2 norm, clipping at 0.2, renormalisation) so the
   descriptor is robust to local illumination and contrast changes.
4. Concatenate all block histograms into one 1-D feature vector.

The implementation wraps ``skimage.feature.hog`` with fixed, documented
constants; it is deterministic and learns nothing from data.
"""

from __future__ import annotations

import numpy as np
from skimage.feature import hog

#: Number of orientation bins over unsigned gradients (0-180 degrees).
HOG_ORIENTATIONS = 9

#: Cell size in pixels: histogramming unit.
HOG_PIXELS_PER_CELL: tuple[int, int] = (8, 8)

#: Block size in cells: normalisation unit (overlapping, stride 1 cell).
HOG_CELLS_PER_BLOCK: tuple[int, int] = (2, 2)

#: Block normalisation scheme (L2 norm, clip, renormalise).
HOG_BLOCK_NORM = "L2-Hys"

#: Whether to apply sqrt compression before gradient computation.
HOG_TRANSFORM_SQRT = False


def hog_feature_dimension(
    image_shape: tuple[int, int],
    pixels_per_cell: tuple[int, int] = HOG_PIXELS_PER_CELL,
    cells_per_block: tuple[int, int] = HOG_CELLS_PER_BLOCK,
    orientations: int = HOG_ORIENTATIONS,
) -> int:
    """Return the HOG vector length for a ``(height, width)`` grayscale image.

    For the project configuration (128x128 image, 8x8 cells, 2x2 blocks,
    9 orientations): 15 x 15 blocks x 4 cells x 9 bins = 8,100 dimensions.

    Raises:
        ValueError: If the image shape is not compatible with the cell/block
            configuration (cells must be at least ``cells_per_block`` wide).
    """
    height, width = image_shape
    if height <= 0 or width <= 0:
        raise ValueError(f"Image shape must be positive, got {image_shape}")
    cells_y = height // pixels_per_cell[1]
    cells_x = width // pixels_per_cell[0]
    if height % pixels_per_cell[1] or width % pixels_per_cell[0]:
        raise ValueError(
            f"Image shape {image_shape} is not divisible by "
            f"pixels_per_cell={pixels_per_cell}"
        )
    blocks_y = cells_y - cells_per_block[1] + 1
    blocks_x = cells_x - cells_per_block[0] + 1
    if blocks_y <= 0 or blocks_x <= 0:
        raise ValueError(
            f"Image shape {image_shape} yields fewer cells than one block "
            f"(cells_per_block={cells_per_block})"
        )
    return int(blocks_y * blocks_x * cells_per_block[0] * cells_per_block[1] * orientations)


def extract_hog(grayscale: np.ndarray) -> np.ndarray:
    """Extract the HOG feature vector from a grayscale image in ``[0, 1]``.

    Args:
        grayscale: ``(H, W)`` float array (the Phase 2 ``grayscale`` array).

    Returns:
        A 1-D float64 vector of fixed length
        :func:`hog_feature_dimension` — finite, deterministic.

    Raises:
        ValueError: If the input is not a non-empty 2-D array or if the
            extracted vector contains non-finite values.
    """
    if grayscale.ndim != 2 or grayscale.size == 0:
        raise ValueError(
            f"Expected a non-empty 2-D grayscale array, got shape {grayscale.shape}"
        )
    features = hog(
        grayscale,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        transform_sqrt=HOG_TRANSFORM_SQRT,
        feature_vector=True,
    )
    features = np.asarray(features, dtype=np.float64)
    if not np.all(np.isfinite(features)):
        raise ValueError("HOG extraction produced non-finite values")
    return features
