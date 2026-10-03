"""Shared pytest fixtures: a tiny synthetic Flickr8k-like dataset.

Unit tests never touch the real Flickr8k archive. They build a few JPEG
images plus the three split text files in a temporary directory so that the
parsing and validation logic can be exercised deterministically anywhere.
Integration tests that require the real dataset are skipped automatically
until it has been placed under ``data/raw/flickr8k/``.
"""

from pathlib import Path

import pytest
from PIL import Image

SYNTHETIC_IMAGE_IDS = [f"synthetic_{index:03d}.jpg" for index in range(1, 6)]


def write_solid_jpeg(path: Path, color=(90, 140, 60)) -> None:
    """Write a small valid JPEG so the file passes PIL's verification."""
    Image.new("RGB", (32, 32), color=color).save(path, format="JPEG")


@pytest.fixture
def clean_dataset(tmp_path: Path) -> dict[str, Path]:
    """A clean miniature Flickr8k: 5 images x 5 captions, 3/1/1 split.

    Returns a mapping with ``raw_dir``, ``images_dir``, ``captions_file``,
    ``train_file``, ``val_file`` and ``test_file`` paths.
    """
    raw_dir = tmp_path / "flickr8k"
    images_dir = raw_dir / "Images"
    images_dir.mkdir(parents=True)
    for image_id in SYNTHETIC_IMAGE_IDS:
        write_solid_jpeg(images_dir / image_id)

    caption_lines = []
    for image_index, image_id in enumerate(SYNTHETIC_IMAGE_IDS, start=1):
        for caption_index in range(5):
            caption_lines.append(
                f"{image_id}#{caption_index}\t"
                f"Synthetic caption {caption_index} about image {image_index} ."
            )
    captions_file = raw_dir / "Flickr8k.token.txt"
    captions_file.write_text("\n".join(caption_lines) + "\n", encoding="utf-8")

    train_file = raw_dir / "Flickr_8k.trainImages.txt"
    train_file.write_text("\n".join(SYNTHETIC_IMAGE_IDS[:3]) + "\n", encoding="utf-8")
    val_file = raw_dir / "Flickr_8k.devImages.txt"
    val_file.write_text(SYNTHETIC_IMAGE_IDS[3] + "\n", encoding="utf-8")
    test_file = raw_dir / "Flickr_8k.testImages.txt"
    test_file.write_text(SYNTHETIC_IMAGE_IDS[4] + "\n", encoding="utf-8")

    return {
        "raw_dir": raw_dir,
        "images_dir": images_dir,
        "captions_file": captions_file,
        "train_file": train_file,
        "val_file": val_file,
        "test_file": test_file,
    }
