"""Phase 3 tests: HOG, colour histogram, LBP and the extraction interface.

Unit tests run against the synthetic Flickr8k-like dataset from conftest.
Integration tests use the real project dataset (skipped when absent) and,
once the dataset-level extraction has been run, verify the saved feature
files for alignment with the Phase 1 CSVs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config, resolve_path
from src.features.extract import extract_features
from src.features.color_histogram import (
    COLOR_BINS_PER_CHANNEL,
    extract_color_histogram,
)
from src.features.hog import extract_hog, hog_feature_dimension
from src.features.lbp import LBP_BINS, extract_lbp
from src.preprocessing.image_preprocessing import TARGET_SIZE, PreprocessedImage, preprocess_image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "flickr8k"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
DATASET_AVAILABLE = (RAW_DIR / "Flickr8k.token.txt").exists()
FEATURES_AVAILABLE = (FEATURES_DIR / "hog_features.npy").exists()

#: Independent expected value for the documented Phase 2 + HOG configuration:
#: (128/8 - 1)^2 blocks x 4 cells x 9 orientations = 8,100.
EXPECTED_HOG_DIMENSION = hog_feature_dimension((TARGET_SIZE[1], TARGET_SIZE[0]))
assert EXPECTED_HOG_DIMENSION == 8100


@pytest.fixture
def preprocessed(clean_dataset) -> PreprocessedImage:
    return preprocess_image(clean_dataset["images_dir"] / "synthetic_001.jpg")


# ---------------------------------------------------------------------------
# HOG
# ---------------------------------------------------------------------------


def test_hog_returns_1d_numpy_array(preprocessed):
    features = extract_hog(preprocessed.grayscale)
    assert isinstance(features, np.ndarray)
    assert features.ndim == 1
    assert features.dtype == np.float64


def test_hog_dimension_is_fixed_and_documented(preprocessed, clean_dataset):
    other = preprocess_image(clean_dataset["images_dir"] / "synthetic_002.jpg")
    first = extract_hog(preprocessed.grayscale)
    assert first.shape == (EXPECTED_HOG_DIMENSION,)
    assert extract_hog(other.grayscale).shape == (EXPECTED_HOG_DIMENSION,)


def test_hog_is_deterministic(preprocessed):
    first = extract_hog(preprocessed.grayscale)
    second = extract_hog(preprocessed.grayscale)
    assert np.array_equal(first, second)


def test_hog_contains_no_nan_or_infinity(preprocessed):
    features = extract_hog(preprocessed.grayscale)
    assert np.isfinite(features).all()


def test_hog_constant_image_has_no_false_edges(preprocessed):
    # A solid-colour image has zero gradients everywhere, so its HOG vector
    # must be identically zero rather than containing numerical noise.
    features = extract_hog(preprocessed.grayscale)
    assert np.allclose(features, 0.0)


def test_hog_rejects_invalid_input():
    with pytest.raises(ValueError):
        extract_hog(np.zeros((0, 0), dtype=np.float32))
    with pytest.raises(ValueError):
        extract_hog(np.zeros((4,), dtype=np.float32))  # 1-D input


# ---------------------------------------------------------------------------
# Colour histogram
# ---------------------------------------------------------------------------


def test_color_histogram_dimension(preprocessed):
    features = extract_color_histogram(preprocessed.hsv)
    assert features.shape == (COLOR_BINS_PER_CHANNEL * 3,)
    assert features.shape == (96,)


def test_color_histogram_l1_normalized_per_channel(preprocessed):
    features = extract_color_histogram(preprocessed.hsv)
    for channel_index, name in enumerate(("H", "S", "V")):
        channel = features[
            channel_index * COLOR_BINS_PER_CHANNEL : (channel_index + 1) * COLOR_BINS_PER_CHANNEL
        ]
        assert np.isclose(channel.sum(), 1.0), name


def test_color_histogram_nonnegative_finite_deterministic(preprocessed):
    first = extract_color_histogram(preprocessed.hsv)
    second = extract_color_histogram(preprocessed.hsv)
    assert np.array_equal(first, second)
    assert (first >= 0.0).all()
    assert np.isfinite(first).all()


def test_color_histogram_rejects_invalid_input():
    with pytest.raises(ValueError):
        extract_color_histogram(np.zeros((0, 0, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        extract_color_histogram(np.zeros((4, 4), dtype=np.float32))


# ---------------------------------------------------------------------------
# LBP
# ---------------------------------------------------------------------------


def test_lbp_dimension(preprocessed):
    features = extract_lbp(preprocessed.grayscale)
    assert features.shape == (LBP_BINS,)
    assert features.shape == (10,)


def test_lbp_histogram_normalized(preprocessed):
    features = extract_lbp(preprocessed.grayscale)
    assert np.isclose(features.sum(), 1.0)
    assert (features >= 0.0).all()


def test_lbp_constant_image_yields_concentrated_histogram(preprocessed):
    # Every interior pixel of a solid-colour image yields the same LBP code;
    # only border pixels differ (scikit-image pads the border). The
    # normalised histogram must therefore be strongly concentrated.
    features = extract_lbp(preprocessed.grayscale)
    assert features.max() > 0.9


def test_lbp_finite_and_deterministic(preprocessed):
    first = extract_lbp(preprocessed.grayscale)
    second = extract_lbp(preprocessed.grayscale)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_lbp_rejects_invalid_input():
    with pytest.raises(ValueError):
        extract_lbp(np.zeros((0, 0), dtype=np.float32))
    with pytest.raises(ValueError):
        extract_lbp(np.zeros((2, 2, 3), dtype=np.float32))  # RGB, not grayscale


# ---------------------------------------------------------------------------
# Combined extraction interface (no fusion — that is Phase 4)
# ---------------------------------------------------------------------------


def test_extract_features_returns_all_three_vectors(preprocessed):
    features = extract_features(preprocessed)
    assert set(features) == {"hog", "color", "lbp"}
    assert features["hog"].shape == (EXPECTED_HOG_DIMENSION,)
    assert features["color"].shape == (96,)
    assert features["lbp"].shape == (10,)
    for vector in features.values():
        assert np.isfinite(vector).all()


# ---------------------------------------------------------------------------
# Integration: real project image
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not DATASET_AVAILABLE,
    reason="Flickr8k not found under data/raw/flickr8k/",
)
def test_real_flickr8k_image_end_to_end():
    captions = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "captions.csv")
    image_id = str(captions["image_id"].iloc[0])
    preprocessed_real = preprocess_image(RAW_DIR / "Images" / image_id, image_id=image_id)
    features = extract_features(preprocessed_real)

    assert features["hog"].shape == (EXPECTED_HOG_DIMENSION,)
    assert features["color"].shape == (96,)
    assert features["lbp"].shape == (10,)
    # a real photograph has edges: its HOG vector must not be all zeros
    assert np.any(features["hog"] != 0.0)
    # three L1-normalised channels sum to 3; the LBP histogram sums to 1
    assert np.isclose(features["color"].sum(), 3.0)
    assert np.isclose(features["lbp"].sum(), 1.0)
    for vector in features.values():
        assert np.isfinite(vector).all()


# ---------------------------------------------------------------------------
# Integration: saved dataset-level feature files
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not FEATURES_AVAILABLE,
    reason="data/features/*.npy not found — run 'python -m src.features.extract' first",
)
class TestSavedFeatureFiles:
    """Verify the saved matrices are complete and aligned with Phase 1."""

    def test_files_are_aligned_with_processed_csvs(self):
        config = load_config()
        image_ids = np.load(FEATURES_DIR / "image_ids.npy")
        splits = np.load(FEATURES_DIR / "splits.npy")
        hog_matrix = np.load(FEATURES_DIR / "hog_features.npy")
        color_matrix = np.load(FEATURES_DIR / "color_features.npy")
        lbp_matrix = np.load(FEATURES_DIR / "lbp_features.npy")
        metadata = json.loads(
            (FEATURES_DIR / "feature_metadata.json").read_text(encoding="utf-8")
        )

        count = len(image_ids)
        assert len(splits) == count
        assert hog_matrix.shape == (count, metadata["hog"]["dimension"])
        assert color_matrix.shape == (count, metadata["color_histogram"]["dimension"])
        assert lbp_matrix.shape == (count, metadata["lbp"]["dimension"])
        assert metadata["project_images"] == count

        captions = pd.read_csv(resolve_path(config["paths"]["processed_dir"]) / "captions.csv")
        expected = captions[["image_id", "split"]].drop_duplicates(
            subset="image_id", keep="first"
        )
        assert [str(value) for value in image_ids] == [str(v) for v in expected["image_id"]]
        assert [str(value) for value in splits] == [str(v) for v in expected["split"]]
        assert set(str(value) for value in np.unique(splits)) <= {
            "train",
            "validation",
            "test",
            "unassigned",
        }
        assert np.isfinite(hog_matrix).all()
        assert np.isfinite(color_matrix).all()
        assert np.isfinite(lbp_matrix).all()

    def test_metadata_records_actual_configuration(self):
        metadata = json.loads(
            (FEATURES_DIR / "feature_metadata.json").read_text(encoding="utf-8")
        )
        assert metadata["hog"]["orientations"] == 9
        assert metadata["hog"]["pixels_per_cell"] == [8, 8]
        assert metadata["hog"]["cells_per_block"] == [2, 2]
        assert metadata["hog"]["block_norm"] == "L2-Hys"
        assert metadata["color_histogram"]["color_space"] == "HSV"
        assert metadata["color_histogram"]["bins_per_channel"] == 32
        assert metadata["color_histogram"]["normalization"].startswith("L1")
        assert metadata["lbp"]["points"] == 8
        assert metadata["lbp"]["radius"] == 1
        assert metadata["lbp"]["method"] == "uniform"
        assert metadata["extraction"]["failed"] == 0
        assert sum(metadata["split_counts"].values()) == metadata["project_images"]
