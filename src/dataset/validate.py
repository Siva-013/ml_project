"""Integrity validation of the parsed Flickr8k dataset.

Two severity levels are used:

* **error** — a violation that would corrupt the downstream pipeline
  (captioned image missing on disk, unreadable image file, duplicate
  caption keys, images listed in more than one split, split entries
  without a caption record or without a file). Errors make
  ``python -m src.dataset`` exit non-zero and block the writing of the
  processed CSV files.
* **warning** — an anomaly worth reporting but not fatal: structurally
  malformed caption lines, empty caption text, images whose caption count
  differs from the expected five, captioned images that belong to no split
  (the official release contains such images by design), and stray files
  in the image directory that have no captions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.dataset.loader import (
    CaptionParseResult,
    find_missing_images,
    find_unreadable_images,
    list_image_files,
)
from src.dataset.split import SPLIT_NAMES, SplitAnalysis

ERROR_CHECKS = (
    "missing_images",
    "corrupt_images",
    "duplicate_caption_keys",
    "split_overlaps",
    "split_duplicates",
    "split_images_missing_from_captions",
    "split_images_missing_on_disk",
)
WARNING_CHECKS = (
    "malformed_caption_lines",
    "empty_caption_rows",
    "captions_per_image_violations",
    "unassigned_images",
    "orphan_image_files",
)


@dataclass
class DatasetValidation:
    """Findings of all Phase 1 dataset integrity checks."""

    missing_images: list[str] = field(default_factory=list)
    corrupt_images: list[str] = field(default_factory=list)
    duplicate_caption_keys: list[str] = field(default_factory=list)
    split_overlaps: dict[str, list[str]] = field(default_factory=dict)
    split_duplicates: dict[str, list[str]] = field(default_factory=dict)
    split_images_missing_from_captions: list[str] = field(default_factory=list)
    split_images_missing_on_disk: list[str] = field(default_factory=list)
    malformed_caption_lines: list[str] = field(default_factory=list)
    empty_caption_rows: list[tuple[str, int]] = field(default_factory=list)
    captions_per_image_violations: dict[str, int] = field(default_factory=dict)
    unassigned_images: list[str] = field(default_factory=list)
    orphan_image_files: list[str] = field(default_factory=list)

    def error_count(self) -> int:
        """Total number of error-severity findings."""
        return (
            len(self.missing_images)
            + len(self.corrupt_images)
            + len(self.duplicate_caption_keys)
            + sum(len(ids) for ids in self.split_overlaps.values())
            + sum(len(ids) for ids in self.split_duplicates.values())
            + len(self.split_images_missing_from_captions)
            + len(self.split_images_missing_on_disk)
        )

    def warning_count(self) -> int:
        """Total number of warning-severity findings."""
        return (
            len(self.malformed_caption_lines)
            + len(self.empty_caption_rows)
            + len(self.captions_per_image_violations)
            + len(self.unassigned_images)
            + len(self.orphan_image_files)
        )

    @property
    def passed(self) -> bool:
        """True when no error-severity finding was raised."""
        return self.error_count() == 0

    def to_dict(self, max_examples: int = 3) -> dict[str, dict]:
        """Serialisable view: per-check severity, count and sample items."""
        severity = {name: "error" for name in ERROR_CHECKS}
        severity.update({name: "warning" for name in WARNING_CHECKS})

        def entry(items: list) -> dict:
            return {
                "count": len(items),
                "examples": [str(item) for item in items[:max_examples]],
                "severity": None,  # filled below
            }

        checks = {
            "missing_images": entry(self.missing_images),
            "corrupt_images": entry(self.corrupt_images),
            "duplicate_caption_keys": entry(self.duplicate_caption_keys),
            "split_overlaps": entry(
                [
                    f"{pair}: {', '.join(ids)}"
                    for pair, ids in sorted(self.split_overlaps.items())
                    if ids
                ]
            ),
            "split_duplicates": entry(
                [
                    f"{name}: {', '.join(ids)}"
                    for name, ids in sorted(self.split_duplicates.items())
                    if ids
                ]
            ),
            "split_images_missing_from_captions": entry(
                self.split_images_missing_from_captions
            ),
            "split_images_missing_on_disk": entry(self.split_images_missing_on_disk),
            "malformed_caption_lines": entry(self.malformed_caption_lines),
            "empty_caption_rows": entry(
                [f"{image_id}#{caption_id}" for image_id, caption_id in self.empty_caption_rows]
            ),
            "captions_per_image_violations": entry(
                [
                    f"{image_id}: {count} captions"
                    for image_id, count in sorted(self.captions_per_image_violations.items())
                ]
            ),
            "unassigned_images": entry(self.unassigned_images),
            "orphan_image_files": entry(self.orphan_image_files),
        }
        for name, item in checks.items():
            item["severity"] = severity[name]
        return checks


def validate_dataset(
    captions: CaptionParseResult,
    splits: SplitAnalysis,
    images_dir: Path,
    *,
    expected_captions_per_image: int = 5,
    verify_image_contents: bool = True,
    image_extensions: Iterable[str] = (".jpg",),
) -> DatasetValidation:
    """Run every Phase 1 integrity check.

    Args:
        captions: Parsed caption file (see :func:`load_caption_file`).
        splits: Parsed split files (see :func:`load_splits`).
        images_dir: Directory containing the raw image files.
        expected_captions_per_image: Reference caption count per image used
            for the captions-per-image warning (5 for Flickr8k).
        verify_image_contents: Whether to open each image file to detect
            corruption (slower, but catches truncated files).
        image_extensions: Recognised image file extensions on disk.

    Returns:
        A :class:`DatasetValidation` with all findings.
    """
    image_extensions = tuple(image_extensions)
    frame = captions.captions
    captioned_ids = set(frame["image_id"])
    on_disk = set(list_image_files(images_dir, image_extensions))

    if frame.empty:
        violations: dict[str, int] = {}
    else:
        counts = frame.groupby("image_id").size()
        violations = {
            image_id: int(count)
            for image_id, count in counts.items()
            if count != expected_captions_per_image
        }

    empty_rows = frame[frame["caption"].fillna("").str.strip() == ""]
    empty_caption_rows = list(zip(empty_rows["image_id"], empty_rows["caption_id"]))

    all_split_ids = splits.all_split_ids()

    return DatasetValidation(
        missing_images=find_missing_images(images_dir, captioned_ids, image_extensions),
        corrupt_images=(
            find_unreadable_images(images_dir, captioned_ids)
            if verify_image_contents
            else []
        ),
        duplicate_caption_keys=list(captions.duplicate_keys),
        split_overlaps={pair: list(ids) for pair, ids in splits.overlaps.items()},
        split_duplicates={
            name: list(ids) for name, ids in splits.duplicates.items()
        },
        split_images_missing_from_captions=sorted(all_split_ids - captioned_ids),
        split_images_missing_on_disk=sorted(all_split_ids - on_disk),
        malformed_caption_lines=list(captions.malformed_lines),
        empty_caption_rows=empty_caption_rows,
        captions_per_image_violations=violations,
        unassigned_images=sorted(captioned_ids - all_split_ids),
        orphan_image_files=sorted(on_disk - captioned_ids),
    )
