"""Phase 2 tests: deterministic image preprocessing.

Unit tests run against the synthetic Flickr8k-like dataset from conftest;
one integration test uses a real project image when the dataset is present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.preprocessing.image_preprocessing import (
    TARGET_SIZE,
    image_to_array,
    load_rgb,
    preprocess_image,
    resize_image,
    to_grayscale,
    to_hsv,
)
from tests.conftest import SYNTHETIC_IMAGE_IDS


# ---------------------------------------------------------------------------
# Unit tests: loading
# ---------------------------------------------------------------------------


def test_load_rgb_returns_rgb_image(clean_dataset):
    image = load_rgb(clean_dataset["images_dir"] / SYNTHETIC_IMAGE_IDS[0])
    assert image.mode == "RGB"


def test_load_rgb_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_rgb(tmp_path / "does_not_exist.jpg")


def test_load_rgb_garbage_file_raises(tmp_path):
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"this is not an image")
    with pytest.raises(ValueError, match="Cannot decode"):
        load_rgb(broken)


# ---------------------------------------------------------------------------
# Unit tests: resizing
# ---------------------------------------------------------------------------


def test_resize_returns_target_size(clean_dataset):
    image = load_rgb(clean_dataset["images_dir"] / SYNTHETIC_IMAGE_IDS[0])
    assert resize_image(image).size == TARGET_SIZE  # (width, height)


def test_resize_rejects_non_positive_size(clean_dataset):
    image = load_rgb(clean_dataset["images_dir"] / SYNTHETIC_IMAGE_IDS[0])
    with pytest.raises(ValueError, match="positive"):
        resize_image(image, size=(0, 10))


# ---------------------------------------------------------------------------
# Unit tests: array conversion and colour spaces
# ---------------------------------------------------------------------------


def test_image_to_array_dtype_shape_and_range(clean_dataset):
    resized = resize_image(load_rgb(clean_dataset["images_dir"] / SYNTHETIC_IMAGE_IDS[0]))
    array = image_to_array(resized)
    assert array.dtype == np.float32
    assert array.shape == (TARGET_SIZE[1], TARGET_SIZE[0], 3)  # (H, W, C)
    assert float(array.min()) >= 0.0
    assert float(array.max()) <= 1.0


def test_to_grayscale_shape_range_and_luma_coefficient():
    red = np.zeros((4, 4, 3), dtype=np.float32)
    red[..., 0] = 1.0
    gray = to_grayscale(red)
    assert gray.shape == (4, 4)
    assert gray.dtype == np.float32
    assert np.allclose(gray, 0.2125, atol=1e-3)  # Rec. 709 luma of pure red
    assert float(gray.min()) >= 0.0 and float(gray.max()) <= 1.0


def test_to_grayscale_rejects_non_rgb_input():
    with pytest.raises(ValueError, match="RGB"):
        to_grayscale(np.zeros((4, 4), dtype=np.float32))


def test_to_grayscale_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        to_grayscale(np.zeros((0, 0, 3), dtype=np.float32))


def test_to_hsv_pure_red_maps_to_h0_s1_v1():
    red = np.zeros((2, 2, 3), dtype=np.float32)
    red[..., 0] = 1.0
    hsv = to_hsv(red)
    assert hsv.shape == (2, 2, 3)
    assert np.allclose(hsv[..., 0], 0.0, atol=1e-6)  # hue
    assert np.allclose(hsv[..., 1], 1.0, atol=1e-6)  # saturation
    assert np.allclose(hsv[..., 2], 1.0, atol=1e-6)  # value


def test_to_hsv_output_in_unit_range(clean_dataset):
    rgb = image_to_array(resize_image(load_rgb(clean_dataset["images_dir"] / SYNTHETIC_IMAGE_IDS[2])))
    hsv = to_hsv(rgb)
    assert float(hsv.min()) >= 0.0
    assert float(hsv.max()) <= 1.0


# ---------------------------------------------------------------------------
# Unit tests: the composed entry point
# ---------------------------------------------------------------------------


def test_preprocess_image_deterministic_and_aligned(clean_dataset):
    path = clean_dataset["images_dir"] / SYNTHETIC_IMAGE_IDS[1]
    first = preprocess_image(path)
    second = preprocess_image(path)

    assert first.image_id == SYNTHETIC_IMAGE_IDS[1]
    assert np.array_equal(first.rgb, second.rgb)
    assert np.array_equal(first.grayscale, second.grayscale)
    assert np.array_equal(first.hsv, second.hsv)

    height, width = TARGET_SIZE[1], TARGET_SIZE[0]
    assert first.rgb.shape == (height, width, 3)
    assert first.grayscale.shape == (height, width)
    assert first.hsv.shape == (height, width, 3)
    assert first.rgb.dtype == np.float32


def test_preprocess_image_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        preprocess_image(tmp_path / "does_not_exist.jpg")


# ---------------------------------------------------------------------------
# Integration test: real project image
# ---------------------------------------------------------------------------


def test_preprocess_real_image():
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw" / "flickr8k"
    captions_path = project_root / "data" / "processed" / "captions.csv"
    if not (raw_dir / "Images").is_dir() or not captions_path.exists():
        pytest.skip("Flickr8k dataset or processed CSVs not present")
    import pandas as pd

    image_id = str(pd.read_csv(captions_path)["image_id"].iloc[0])
    preprocessed = preprocess_image(raw_dir / "Images" / image_id, image_id=image_id)
    assert preprocessed.rgb.shape == (TARGET_SIZE[1], TARGET_SIZE[0], 3)
    assert 0.0 <= float(preprocessed.grayscale.min())
    assert float(preprocessed.grayscale.max()) <= 1.0
