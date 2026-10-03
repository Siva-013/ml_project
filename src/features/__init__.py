"""Phase 3 — classical visual feature extraction (HOG, colour histogram, LBP).

The combined per-image interface ``extract_features`` lives in
``src.features.extract`` (not re-exported here so that ``python -m
src.features.extract`` runs cleanly).
"""

from src.features.color_histogram import extract_color_histogram
from src.features.hog import extract_hog
from src.features.lbp import extract_lbp

__all__ = ["extract_hog", "extract_color_histogram", "extract_lbp"]
