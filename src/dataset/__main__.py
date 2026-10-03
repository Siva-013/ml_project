"""Phase 1 pipeline: load, validate, split and report on the Flickr8k dataset.

Run from the project root::

    python -m src.dataset [--config configs/config.yaml]

Image IDs listed in ``dataset.excluded_images`` (configs/config.yaml) are
excluded from the project dataset at load time, with their documented
reasons; the raw source files are never modified. The exclusion is applied
before validation, so the validation rules remain strict for everything the
project dataset contains.

Exit codes:
    0 — validation passed, processed CSV files were written.
    1 — validation failed; reports were written, CSV files were NOT written.
    2 — required input files are missing or the configuration is invalid.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import load_config, resolve_path
from src.dataset import report
from src.dataset.exclusions import ExclusionResult, apply_exclusions, parse_exclusions
from src.dataset.loader import CaptionParseResult, load_caption_file
from src.dataset.split import assign_splits, load_splits, write_processed_csvs
from src.dataset.validate import validate_dataset


def _apply_configured_exclusions(
    captions: CaptionParseResult,
    exclusions_raw: list[dict] | None,
    images_dir: Path,
) -> tuple[CaptionParseResult | None, dict | None]:
    """Apply ``dataset.excluded_images`` and build its report block.

    Returns:
        ``(captions, exclusions_block)``. ``captions`` is ``None`` when the
        configuration is invalid (an exclusion ID absent from the caption
        file) after the error has been printed; the caller then aborts with
        exit code 2. ``exclusions_block`` is ``None`` when nothing is
        configured.
    """
    exclusions = parse_exclusions(exclusions_raw)
    if not exclusions:
        return captions, None

    exclusion: ExclusionResult = apply_exclusions(captions, exclusions)
    if exclusion.unknown_ids:
        print(
            "[error] dataset.excluded_images references image IDs that do not "
            "appear in the caption file:",
            file=sys.stderr,
        )
        for image_id in exclusion.unknown_ids:
            print(f"        - {image_id}", file=sys.stderr)
        return None, None

    counts = exclusion.removed.groupby("image_id").size()
    print(
        f"      project exclusions: {len(exclusion.removed_ids)} image ID(s), "
        f"{len(exclusion.removed)} caption row(s) removed "
        "(raw source files untouched)"
    )

    stale_ids = [
        entry.image_id
        for entry in exclusions
        if (images_dir / entry.image_id).exists()
    ]
    if stale_ids:
        print(
            "[warning] an image file now exists for excluded ID(s): "
            f"{', '.join(stale_ids)} — revisit the exclusion",
            file=sys.stderr,
        )

    exclusions_block = {
        "mechanism": (
            "dataset.excluded_images in configs/config.yaml; "
            "the raw source files are never modified"
        ),
        "entries": [
            {
                "image_id": entry.image_id,
                "reason": entry.reason,
                "captions_removed": int(counts.get(entry.image_id, 0)),
            }
            for entry in exclusions
        ],
        "removed_captions_total": int(len(exclusion.removed)),
        "stale_ids": stale_ids,
    }
    return exclusion.remaining, exclusions_block


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.dataset",
        description="Phase 1: load, validate and split the Flickr8k dataset.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml (default: configs/config.yaml in the project root)",
    )
    return parser.parse_args(argv)


def _required_input_paths(config: dict) -> dict[str, Path]:
    paths = config["paths"]
    return {
        "caption file": resolve_path(paths["captions_file"]),
        "images directory": resolve_path(paths["images_dir"]),
        "train split file": resolve_path(paths["train_split_file"]),
        "validation split file": resolve_path(paths["val_split_file"]),
        "test split file": resolve_path(paths["test_split_file"]),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as error:
        print(f"[error] {error}", file=sys.stderr)
        return 2

    inputs = _required_input_paths(config)
    missing_inputs = [(name, path) for name, path in inputs.items() if not path.exists()]
    if missing_inputs:
        print("[error] Required dataset files are missing:", file=sys.stderr)
        for name, path in missing_inputs:
            print(f"        - {name}: {path}", file=sys.stderr)
        print(
            "        Download Flickr8k and place it under data/raw/flickr8k/ "
            "(see README.md, section 'Dataset acquisition').",
            file=sys.stderr,
        )
        return 2

    dataset_config = config.get("dataset", {})
    validation_config = config.get("validation", {})
    report_config = config.get("report", {})
    expected = int(dataset_config.get("expected_captions_per_image", 5))
    extensions = tuple(dataset_config.get("image_extensions", [".jpg"]))
    verify_contents = bool(validation_config.get("verify_image_contents", True))

    print("[1/6] Parsing caption file ...")
    captions = load_caption_file(inputs["caption file"])
    print(
        f"      {len(captions.captions)} caption rows, "
        f"{captions.captions['image_id'].nunique()} unique images, "
        f"{len(captions.malformed_lines)} malformed line(s), "
        f"{len(captions.duplicate_keys)} duplicate key(s)"
    )
    source_totals = {
        "referenced_image_ids": int(captions.captions["image_id"].nunique()),
        "captions": int(len(captions.captions)),
    }

    captions, exclusions_block = _apply_configured_exclusions(
        captions, dataset_config.get("excluded_images"), inputs["images directory"]
    )
    if captions is None:
        return 2

    print("[2/6] Loading official split files ...")
    splits = load_splits(
        inputs["train split file"],
        inputs["validation split file"],
        inputs["test split file"],
    )
    print(
        f"      train={len(splits.train)}, "
        f"validation={len(splits.validation)}, "
        f"test={len(splits.test)} images"
    )

    print("[3/6] Validating dataset ...")
    validation = validate_dataset(
        captions,
        splits,
        inputs["images directory"],
        expected_captions_per_image=expected,
        verify_image_contents=verify_contents,
        image_extensions=extensions,
    )
    for name, item in validation.to_dict().items():
        if item["count"]:
            print(f"      [{item['severity']}] {name}: {item['count']}")

    captions_with_split = assign_splits(captions.captions, splits)

    written_csvs = None
    if validation.passed:
        print("[4/6] Writing processed CSV files ...")
        written_csvs = write_processed_csvs(
            captions_with_split, resolve_path(config["paths"]["processed_dir"])
        )
    else:
        print("[4/6] SKIPPED writing processed CSV files (validation failed).")

    print("[5/6] Building dataset report ...")
    stats = report.build_statistics(
        captions_with_split,
        validation,
        inputs["images directory"],
        expected_captions_per_image=expected,
        image_extensions=extensions,
        source_totals=source_totals,
        exclusions=exclusions_block,
    )
    report_paths: list[Path] = []
    if report_config.get("json_report"):
        report_paths.append(
            report.write_json_report(stats, resolve_path(report_config["json_report"]))
        )
    if report_config.get("markdown_report"):
        report_paths.append(
            report.write_markdown_report(
                stats, resolve_path(report_config["markdown_report"])
            )
        )

    print("[6/6] Done.")
    report.print_summary(stats, written_csvs=written_csvs, report_paths=report_paths)
    return 0 if validation.passed else 1


if __name__ == "__main__":
    sys.exit(main())
