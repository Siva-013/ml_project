"""Phase 3 — Local Binary Pattern texture features (classical, deterministic).

LBP captures local texture by comparing each pixel with its circular
neighbourhood:

1. For every pixel, sample ``P`` neighbours on a circle of radius ``R``.
2. Threshold each neighbour by the centre pixel: neighbour >= centre -> 1,
   else 0.
3. Read the resulting bits as one integer LBP code (per pixel).
4. With ``method="uniform"``: patterns with at most two 0/1 transitions are
   "uniform" and keep their own label; all other patterns are pooled into a
   single extra label. This gives ``P + 2`` possible values (here 8 + 2 = 10)
   and makes the descriptor robust to rotation and noise.
5. Build the L1-normalised histogram of LBP codes over the whole image so
   the feature magnitude does not depend on image size.

The implementation wraps ``skimage.feature.local_binary_pattern`` with fixed,
documented constants; it is deterministic and learns nothing from data.
"""

from __future__ import annotations

import numpy as np
from skimage.feature import local_binary_pattern

#: Number of neighbours on the circle.
LBP_POINTS = 8

#: Circle radius in pixels.
LBP_RADIUS = 1

#: Uniform mapping (<= 2 transitions keep their label, others pooled).
LBP_METHOD = "uniform"

#: Histogram bins for uniform LBP: P + 2.
LBP_BINS = LBP_POINTS + 2


def quantise_grayscale(grayscale: np.ndarray) -> np.ndarray:
    """Quantise a ``[0, 1]`` float grayscale image to 8-bit unsigned luma.

    Shared by :func:`extract_lbp` and the visualisation utility so both
    operate on exactly the same integer pixel values.
    """
    return np.clip(np.round(grayscale * 255.0), 0, 255).astype(np.uint8)


def extract_lbp(grayscale: np.ndarray) -> np.ndarray:
    """Extract the L1-normalised uniform-LBP histogram of a grayscale image.

    The input is quantised to 8-bit integer luma (``round(x * 255)``)
    before the LBP codes are computed. This follows the classical LBP
    definition — thresholding of *integer* pixel neighbours — and avoids the
    float-comparison artefacts scikit-image warns about for floating-point
    images. Quantisation is deterministic.

    Args:
        grayscale: ``(H, W)`` float array in ``[0, 1]`` (the Phase 2
            ``grayscale`` array).

    Returns:
        A 1-D float64 vector of length ``P + 2`` = 10, summing to 1.
        Finite and deterministic. Note that border pixels see fewer valid
        neighbours (scikit-image pads the border), so a constant image
        yields a strongly concentrated — but not perfectly single-bin —
        histogram.

    Raises:
        ValueError: If the input is not a non-empty 2-D array, or if the
            histogram contains non-finite values.
    """
    if grayscale.ndim != 2 or grayscale.size == 0:
        raise ValueError(
            f"Expected a non-empty 2-D grayscale array, got shape {grayscale.shape}"
        )
    codes = local_binary_pattern(
        quantise_grayscale(grayscale), P=LBP_POINTS, R=LBP_RADIUS, method=LBP_METHOD
    )
    counts, _ = np.histogram(codes, bins=LBP_BINS, range=(0, LBP_BINS))
    total = counts.sum()
    if total == 0:
        raise ValueError("LBP received an image with no pixels")
    features = counts.astype(np.float64) / total
    if not np.all(np.isfinite(features)):
        raise ValueError("LBP produced non-finite values")
    return features
