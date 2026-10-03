"""Dataset statistics and the Phase 1 validation report.

Every number in the report is computed from the files actually present on
disk at preparation time; no dataset statistic is hard-coded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.dataset.loader import list_image_files
from src.dataset.split import SPLIT_NAMES
from src.dataset.validate import DatasetValidation

UNASSIGNED = "unassigned"


def build_statistics(
    captions_with_split: pd.DataFrame,
    validation: DatasetValidation,
    images_dir: Path,
    *,
    expected_captions_per_image: int = 5,
    image_extensions=(".jpg",),
    source_totals: dict | None = None,
    exclusions: dict | None = None,
) -> dict:
    """Compute all Phase 1 dataset statistics.

    Args:
        captions_with_split: Caption rows with the ``split`` column attached
            (the project dataset, i.e. after any configured exclusions).
        validation: Validation findings to embed in the report.
        images_dir: Raw image directory (counted from disk).
        expected_captions_per_image: Expected reference count per image.
        image_extensions: Recognised image file extensions on disk.
        source_totals: Optional statistics of the unmodified raw source
            (``referenced_image_ids`` / ``captions``), embedded as
            ``original_source`` so the report distinguishes the original
            Flickr8k files from the project dataset.
        exclusions: Optional exclusions block (mechanism, entries with
            reasons, removed-caption counts, stale IDs), embedded as
            ``exclusions``.

    Returns:
        A JSON-serialisable statistics dictionary.
    """
    frame = captions_with_split
    per_image = frame.groupby("image_id").size() if not frame.empty else pd.Series(dtype="int64")

    splits_block: dict[str, dict[str, int]] = {}
    for split in SPLIT_NAMES + (UNASSIGNED,):
        subset = frame[frame["split"] == split]
        splits_block[split] = {
            "images": int(subset["image_id"].nunique()),
            "captions": int(len(subset)),
        }

    if per_image.empty:
        captions_per_image = {"minimum": 0, "maximum": 0, "mean": 0.0}
    else:
        captions_per_image = {
            "minimum": int(per_image.min()),
            "maximum": int(per_image.max()),
            "mean": round(float(per_image.mean()), 2),
        }
    captions_per_image["expected"] = int(expected_captions_per_image)

    stats = {
        "totals": {
            "images_on_disk": int(len(list_image_files(images_dir, image_extensions))),
            "images_with_captions": int(per_image.shape[0]),
            "total_captions": int(len(frame)),
        },
        "captions_per_image": captions_per_image,
        "splits": splits_block,
        "validation": {
            "passed": bool(validation.passed),
            "error_count": int(validation.error_count()),
            "warning_count": int(validation.warning_count()),
        },
        "issues": validation.to_dict(),
    }
    if source_totals is not None:
        stats["original_source"] = dict(source_totals)
    if exclusions is not None:
        stats["exclusions"] = dict(exclusions)
    return stats


def write_json_report(stats: dict, path: Path) -> Path:
    """Write the statistics dictionary as an indented JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, default=str)
    return path


def _escape(value: str) -> str:
    return value.replace("|", "\\|")


def _exclusion_sections(stats: dict) -> list[str]:
    """Markdown sections distinguishing the original source from the project dataset.

    Rendered only when the pipeline ran with configured exclusions; without
    them the report is identical to the plain dataset report.
    """
    sections: list[str] = []
    if "exclusions" in stats:
        sections += [
            "These totals describe the **project dataset after the "
            "intentional exclusion documented below** — not the complete "
            "original Flickr8k dataset.",
            "",
        ]
    if "original_source" in stats:
        source = stats["original_source"]
        sections += [
            "## Original Flickr8k source (raw files, unmodified)",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Image IDs referenced by Flickr8k.token.txt | {source['referenced_image_ids']} |",
            f"| Caption rows in Flickr8k.token.txt | {source['captions']} |",
            "",
        ]
    if "exclusions" in stats:
        block = stats["exclusions"]
        sections += [
            "## Intentional project exclusions",
            "",
            f"Mechanism: {block['mechanism']}.",
            "",
            "| Image ID | Captions removed | Reason |",
            "| --- | ---: | --- |",
        ]
        for entry in block["entries"]:
            sections.append(
                f"| {_escape(entry['image_id'])} | {entry['captions_removed']} "
                f"| {_escape(entry['reason'])} |"
            )
        sections += [
            f"| **Total** | **{block['removed_captions_total']}** | |",
            "",
        ]
        if block["stale_ids"]:
            sections += [
                "> Warning: an image file now exists for "
                f"{', '.join(block['stale_ids'])} — revisit the exclusion.",
                "",
            ]
        sections += [
            "The project dataset therefore contains "
            f"{stats['totals']['images_with_captions']} captioned images and "
            f"{stats['totals']['total_captions']} captions. It is a documented "
            "derivative of Flickr8k, not the complete original dataset.",
            "",
        ]
    return sections


def write_markdown_report(stats: dict, path: Path) -> Path:
    """Render the statistics dictionary as a human-readable Markdown report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    totals = stats["totals"]
    cpi = stats["captions_per_image"]
    verdict = stats["validation"]
    extra_sections = _exclusion_sections(stats)

    lines = [
        "# Flickr8k Dataset Validation Report (Phase 1)",
        "",
        "All values below are computed from the raw files present in "
        "`data/raw/flickr8k/` at preparation time; nothing is hard-coded.",
        "",
        "## Totals",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Images on disk | {totals['images_on_disk']} |",
        f"| Images with captions | {totals['images_with_captions']} |",
        f"| Total captions | {totals['total_captions']} |",
        *extra_sections,
        "## Captions per image",
        "",
        "| Expected | Min | Max | Mean |",
        "| ---: | ---: | ---: | ---: |",
        f"| {cpi['expected']} | {cpi['minimum']} | {cpi['maximum']} | {cpi['mean']} |",
        "",
        "## Splits",
        "",
        "| Split | Images | Captions |",
        "| --- | ---: | ---: |",
    ]
    for split, block in stats["splits"].items():
        lines.append(f"| {split} | {block['images']} | {block['captions']} |")

    lines += [
        "",
        "## Validation checks",
        "",
        "| Check | Severity | Count | Examples |",
        "| --- | --- | ---: | --- |",
    ]
    for name, item in stats["issues"].items():
        examples = "; ".join(_escape(example) for example in item["examples"])
        lines.append(
            f"| {name} | {item['severity']} | {item['count']} | {examples or '—'} |"
        )

    status = "PASSED" if verdict["passed"] else "FAILED"
    lines += [
        "",
        f"**Validation result: {status}** — {verdict['error_count']} error(s), "
        f"{verdict['warning_count']} warning(s).",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def print_summary(
    stats: dict,
    written_csvs: dict[str, Path] | None = None,
    report_paths: list[Path] | None = None,
) -> None:
    """Print a compact human-readable summary to stdout."""
    totals = stats["totals"]
    cpi = stats["captions_per_image"]
    verdict = stats["validation"]
    bar = "=" * 64

    print(bar)
    print("Phase 1 — Flickr8k dataset summary (computed from raw files)")
    print(bar)
    if "original_source" in stats:
        source = stats["original_source"]
        print(
            f"Original Flickr8k source         : {source['referenced_image_ids']} "
            f"image IDs, {source['captions']} captions (raw files, unmodified)"
        )
    if "exclusions" in stats:
        block = stats["exclusions"]
        print(
            f"Intentional exclusions           : {len(block['entries'])} image ID(s), "
            f"{block['removed_captions_total']} captions (dataset.excluded_images)"
        )
        print("Project dataset (after documented exclusion):")
    print(f"Images on disk                   : {totals['images_on_disk']}")
    print(f"Images with captions             : {totals['images_with_captions']}")
    print(f"Total captions                   : {totals['total_captions']}")
    print(
        f"Captions per image (min/max/mean) : "
        f"{cpi['minimum']} / {cpi['maximum']} / {cpi['mean']}"
    )
    for split, block in stats["splits"].items():
        print(
            f"{split:<12}: {block['images']:>6} images, {block['captions']:>6} captions"
        )
    status = "PASSED" if verdict["passed"] else "FAILED"
    print(
        f"Validation                       : {status} "
        f"({verdict['error_count']} errors, {verdict['warning_count']} warnings)"
    )
    if written_csvs:
        print("Processed CSV files:")
        for name, path in written_csvs.items():
            print(f"  - {name:<12}: {path}")
    if report_paths:
        print("Reports:")
        for path in report_paths:
            print(f"  - {path}")
    print(bar)
