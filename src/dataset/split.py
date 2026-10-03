"""Official train/validation/test split handling for Flickr8k.

The official distribution ships three text files, each listing one image
filename per line: ``Flickr_8k.trainImages.txt``,
``Flickr_8k.devImages.txt`` (validation) and ``Flickr_8k.testImages.txt``.
This module reads them, computes integrity information (duplicates within a
file, overlaps between files) and attaches the split label to every caption
row. The official split is used as-is; the project never re-splits the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import pandas as pd

from src.dataset.loader import ImageListResult, read_image_list

SPLIT_NAMES = ("train", "validation", "test")
PROCESSED_FILENAMES = {
    "captions": "captions.csv",
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv",
}
UNASSIGNED_LABEL = "unassigned"


@dataclass
class SplitAnalysis:
    """Parsed split files plus integrity information.

    Attributes:
        train: Unique train image filenames, in file order.
        validation: Unique validation image filenames, in file order.
        test: Unique test image filenames, in file order.
        duplicates: Filenames listed more than once, per split file.
        overlaps: Filenames shared between two splits, keyed by
            ``"<split_a>_and_<split_b>"``.
    """

    train: list[str] = field(default_factory=list)
    validation: list[str] = field(default_factory=list)
    test: list[str] = field(default_factory=list)
    duplicates: dict[str, list[str]] = field(default_factory=dict)
    overlaps: dict[str, list[str]] = field(default_factory=dict)

    def names_for(self, split: str) -> list[str]:
        """Return the image filenames belonging to ``split``."""
        if split not in SPLIT_NAMES:
            raise ValueError(f"Unknown split {split!r}; expected one of {SPLIT_NAMES}")
        return getattr(self, split)

    def all_split_ids(self) -> set[str]:
        """Return the union of all images assigned to any split."""
        union: set[str] = set()
        for split in SPLIT_NAMES:
            union.update(self.names_for(split))
        return union


def load_splits(
    train_path: Path, validation_path: Path, test_path: Path
) -> SplitAnalysis:
    """Read the three official split files and compute overlap/duplicate info.

    Raises:
        FileNotFoundError: If any of the three files does not exist.
    """
    paths = {
        "train": train_path,
        "validation": validation_path,
        "test": test_path,
    }
    results: dict[str, ImageListResult] = {
        name: read_image_list(path) for name, path in paths.items()
    }

    overlaps: dict[str, list[str]] = {}
    for first, second in combinations(SPLIT_NAMES, 2):
        pair = f"{first}_and_{second}"
        overlaps[pair] = sorted(set(results[first].names) & set(results[second].names))

    return SplitAnalysis(
        train=results["train"].names,
        validation=results["validation"].names,
        test=results["test"].names,
        duplicates={name: result.duplicates for name, result in results.items()},
        overlaps=overlaps,
    )


def assign_splits(captions: pd.DataFrame, splits: SplitAnalysis) -> pd.DataFrame:
    """Return a copy of ``captions`` with a ``split`` column added.

    Images absent from every split file receive the label ``unassigned``;
    the official Flickr8k release contains such images by design (more
    captioned images than split-assigned images).
    """
    label_by_image = {
        image_id: split
        for split in SPLIT_NAMES
        for image_id in splits.names_for(split)
    }
    result = captions.copy()
    result["split"] = result["image_id"].map(label_by_image).fillna(UNASSIGNED_LABEL)
    return result


def write_processed_csvs(
    captions_with_split: pd.DataFrame, processed_dir: Path
) -> dict[str, Path]:
    """Write ``captions.csv``, ``train.csv``, ``validation.csv``, ``test.csv``.

    Rows are sorted by ``(image_id, caption_id)`` for deterministic output.
    Only rows labelled train/validation/test go into the per-split files;
    unassigned rows remain in ``captions.csv`` only.

    Returns:
        Mapping from file role (``captions``/``train``/``validation``/``test``)
        to the written path.
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    ordered = captions_with_split.sort_values(
        ["image_id", "caption_id"], kind="stable"
    ).reset_index(drop=True)

    written: dict[str, Path] = {}
    written["captions"] = processed_dir / PROCESSED_FILENAMES["captions"]
    ordered.to_csv(written["captions"], index=False)
    for split in SPLIT_NAMES:
        written[split] = processed_dir / PROCESSED_FILENAMES[split]
        ordered[ordered["split"] == split].to_csv(written[split], index=False)
    return written
