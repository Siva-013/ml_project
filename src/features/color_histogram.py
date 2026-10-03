"""Phase 3 — colour histogram features (classical, deterministic).

The colour histogram captures the global colour distribution of an image,
independent of where objects appear:

1. Convert the image to HSV, which separates colour content (hue) from its
   purity (saturation) and brightness (value), making the descriptor more
   robust to lighting variation than raw RGB.
2. Bin each channel into 32 uniform bins over ``[0, 1]``.
3. L1-normalise each channel histogram (sums to 1) so the feature magnitude
   does not depend on image size.
4. Concatenate ``H + S + V`` into one vector of 3 x 32 = 96 dimensions.

The implementation is deterministic and learns nothing from data.
"""

from __future__ import annotations

import numpy as np

#: Colour space used for the histogram.
COLOR_SPACE = "HSV"

#: Histogram bins per HSV channel.
COLOR_BINS_PER_CHANNEL = 32

#: Value range of each HSV channel produced by Phase 2 preprocessing.
COLOR_BIN_RANGE: tuple[float, float] = (0.0, 1.0)

#: Normalisation applied to each channel histogram.
COLOR_NORMALIZATION = "L1 (per channel)"

#: Dimension of the final feature vector (channels x bins).
COLOR_DIMENSION = 3 * COLOR_BINS_PER_CHANNEL

_CHANNEL_NAMES = ("H", "S", "V")


def extract_color_histogram(hsv: np.ndarray) -> np.ndarray:
    """Extract the concatenated, per-channel L1-normalised HSV histogram.

    Args:
        hsv: ``(H, W, 3)`` float array in ``[0, 1]`` (the Phase 2 ``hsv``
            array, channel order H, S, V).

    Returns:
        A 1-D float64 vector of length 96: ``[H histogram, S histogram,
        V histogram]``, each channel's 32 bins summing to 1. Finite and
        deterministic.

    Raises:
        ValueError: If the input is not a non-empty ``(H, W, 3)`` array or
            if the resulting vector contains non-finite values.
    """
    if hsv.ndim != 3 or hsv.shape[2] != 3 or hsv.size == 0:
        raise ValueError(f"Expected a non-empty HSV array of shape (H, W, 3), got {hsv.shape}")

    channel_histograms = []
    for channel_index in range(3):
        counts, _ = np.histogram(
            hsv[:, :, channel_index],
            bins=COLOR_BINS_PER_CHANNEL,
            range=COLOR_BIN_RANGE,
        )
        total = counts.sum()
        if total == 0:
            raise ValueError("Colour histogram received an image with no pixels")
        channel_histograms.append(counts.astype(np.float64) / total)

    features = np.concatenate(channel_histograms)
    if not np.all(np.isfinite(features)):
        raise ValueError("Colour histogram produced non-finite values")
    return features
