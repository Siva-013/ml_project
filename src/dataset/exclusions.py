"""Project-level exclusion of known-defective official dataset entries.

The raw Flickr8k files are treated as read-only source material and are
never modified. Known upstream defects — for example an image ID that the
official ``Flickr8k.token.txt`` references but the official image archive
does not contain — are excluded through configuration
(``dataset.excluded_images`` in ``configs/config.yaml``), applied at load
time, and reported in every run and in the dataset report. The evidence and
the decision record live in ``docs/phase1_dataset.md`` (appendix, sections
12-13).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.dataset.loader import CaptionParseResult


@dataclass(frozen=True)
class ExcludedImage:
    """One configured exclusion: an image ID and its documented reason."""

    image_id: str
    reason: str


@dataclass
class ExclusionResult:
    """Outcome of applying the configured exclusions to the caption table.

    Attributes:
        remaining: The caption rows kept by the project dataset (original
            text and order).
        removed: The removed caption rows, verbatim, for reporting.
        removed_ids: Configured IDs that were present and removed.
        unknown_ids: Configured IDs absent from the caption file (config
            drift); the caller decides how to surface them.
    """

    remaining: CaptionParseResult
    removed: pd.DataFrame
    removed_ids: list[str] = field(default_factory=list)
    unknown_ids: list[str] = field(default_factory=list)


def parse_exclusions(raw: list[dict] | None) -> list[ExcludedImage]:
    """Validate and convert the ``dataset.excluded_images`` config entries.

    Args:
        raw: The raw config list (``None``/empty means no exclusions).

    Returns:
        Validated :class:`ExcludedImage` entries.

    Raises:
        ValueError: If an entry is not a mapping with a non-empty ``id`` and
            a non-empty ``reason``, or if the same ID is listed twice.
    """
    if not raw:
        return []
    if not isinstance(raw, list):
        raise ValueError(
            "dataset.excluded_images must be a list of {id, reason} entries"
        )
    entries: list[ExcludedImage] = []
    seen: set[str] = set()
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict) or not item.get("id") or not item.get("reason"):
            raise ValueError(
                f"dataset.excluded_images entry #{index} must be a mapping "
                "with non-empty 'id' and 'reason' keys"
            )
        image_id = str(item["id"]).strip()
        if not image_id:
            raise ValueError(f"dataset.excluded_images entry #{index} has an empty id")
        if image_id in seen:
            raise ValueError(
                f"dataset.excluded_images lists {image_id!r} more than once"
            )
        seen.add(image_id)
        entries.append(
            ExcludedImage(image_id=image_id, reason=str(item["reason"]).strip())
        )
    return entries


def apply_exclusions(
    captions: CaptionParseResult, exclusions: list[ExcludedImage]
) -> ExclusionResult:
    """Remove every caption row belonging to the excluded image IDs.

    Remaining rows keep their original order and text. Removed rows are
    returned verbatim so reports can document exactly what was excluded.
    The validator is not involved: exclusion happens before validation, so
    validation rules stay strict for everything the project dataset
    actually contains.
    """
    if not exclusions:
        return ExclusionResult(remaining=captions, removed=captions.captions.iloc[0:0].copy())

    excluded_ids = [entry.image_id for entry in exclusions]
    mask = captions.captions["image_id"].isin(excluded_ids)
    removed = captions.captions[mask].copy()
    remaining_frame = captions.captions[~mask].copy().reset_index(drop=True)
    present_ids = set(captions.captions["image_id"])
    return ExclusionResult(
        remaining=CaptionParseResult(
            captions=remaining_frame,
            malformed_lines=list(captions.malformed_lines),
            duplicate_keys=list(captions.duplicate_keys),
        ),
        removed=removed,
        removed_ids=[image_id for image_id in excluded_ids if image_id in present_ids],
        unknown_ids=[
            image_id for image_id in excluded_ids if image_id not in present_ids
        ],
    )
