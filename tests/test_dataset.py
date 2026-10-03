"""Phase 1 tests: parsing, validation, splitting, reports and the pipeline.

Two layers:

* **Unit tests** run against a tiny synthetic Flickr8k-like dataset built in
  a temporary directory — they always run and cover edge cases (malformed
  lines, missing/corrupt images, overlapping splits, unassigned images).
* **Integration tests** run only when the real Flickr8k archive is present
  under ``data/raw/flickr8k/``; they are skipped otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.config import load_config
from src.dataset import report
from src.dataset.__main__ import main as phase1_main
from src.dataset.exclusions import apply_exclusions, parse_exclusions
from src.dataset.loader import (
    find_missing_images,
    find_unreadable_images,
    list_image_files,
    load_caption_file,
    read_image_list,
)
from src.dataset.split import assign_splits, load_splits, write_processed_csvs
from src.dataset.validate import validate_dataset
from tests.conftest import SYNTHETIC_IMAGE_IDS, write_solid_jpeg

# ---------------------------------------------------------------------------
# Real-dataset integration marker
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "flickr8k"
REAL_DATASET_AVAILABLE = (RAW_DIR / "Flickr8k.token.txt").exists()

requires_real_dataset = pytest.mark.skipif(
    not REAL_DATASET_AVAILABLE,
    reason="Flickr8k not found under data/raw/flickr8k/ — see README.md "
    "('Dataset acquisition')",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prepare(dataset: dict[str, Path]):
    """Parse caption + split files of a synthetic dataset."""
    captions = load_caption_file(dataset["captions_file"])
    splits = load_splits(dataset["train_file"], dataset["val_file"], dataset["test_file"])
    return captions, splits


def _write_config(dataset: dict[str, Path], tmp_path: Path):
    """Write a config.yaml pointing at a synthetic dataset."""
    processed_dir = tmp_path / "processed"
    report_dir = tmp_path / "reports"
    config = {
        "paths": {
            "raw_data_dir": str(dataset["raw_dir"]),
            "images_dir": str(dataset["images_dir"]),
            "captions_file": str(dataset["captions_file"]),
            "train_split_file": str(dataset["train_file"]),
            "val_split_file": str(dataset["val_file"]),
            "test_split_file": str(dataset["test_file"]),
            "processed_dir": str(processed_dir),
        },
        "dataset": {"expected_captions_per_image": 5, "image_extensions": [".jpg"]},
        "validation": {"verify_image_contents": True},
        "report": {
            "json_report": str(report_dir / "dataset_report.json"),
            "markdown_report": str(report_dir / "dataset_report.md"),
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, processed_dir


# ---------------------------------------------------------------------------
# Unit tests: caption file parsing
# ---------------------------------------------------------------------------


def test_caption_file_parses_all_rows(clean_dataset):
    result = load_caption_file(clean_dataset["captions_file"])
    assert list(result.captions.columns) == ["image_id", "caption_id", "caption"]
    assert len(result.captions) == 25
    for _, group in result.captions.groupby("image_id"):
        assert set(group["caption_id"]) == {0, 1, 2, 3, 4}
    assert result.malformed_lines == []
    assert result.duplicate_keys == []


def test_caption_file_preserves_text_verbatim(tmp_path):
    token = tmp_path / "token.txt"
    original = "A Dog, running! (fast) ."
    token.write_text(f"pic.jpg#0\t{original}\n", encoding="utf-8")
    row = load_caption_file(token).captions.iloc[0]
    assert row["image_id"] == "pic.jpg"
    assert row["caption_id"] == 0
    assert row["caption"] == original


def test_caption_file_reports_malformed_lines(tmp_path):
    token = tmp_path / "token.txt"
    token.write_text(
        "pic.jpg#0\ta caption\n"
        "no tab in this line\n"
        "pic.jpg/x\tanother caption\n"
        "\n"
        "pic.jpg#1\tok caption\n",
        encoding="utf-8",
    )
    result = load_caption_file(token)
    assert len(result.captions) == 2
    assert len(result.malformed_lines) == 2
    assert all("line" in message for message in result.malformed_lines)


def test_caption_file_deduplicates_repeated_keys(tmp_path):
    token = tmp_path / "token.txt"
    token.write_text(
        "a.jpg#0\tfirst\na.jpg#0\tsecond\na.jpg#1\tthird\n", encoding="utf-8"
    )
    result = load_caption_file(token)
    assert len(result.captions) == 2
    assert result.captions.iloc[0]["caption"] == "first"
    assert result.duplicate_keys == ["a.jpg#0"]


def test_caption_file_keeps_empty_caption_rows(tmp_path):
    token = tmp_path / "token.txt"
    token.write_text("a.jpg#0\t   \n", encoding="utf-8")
    result = load_caption_file(token)
    assert len(result.captions) == 1
    assert result.captions.iloc[0]["caption"] == ""
    assert result.malformed_lines == []


def test_caption_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_caption_file(tmp_path / "does_not_exist.txt")


# ---------------------------------------------------------------------------
# Unit tests: split file parsing and image directory
# ---------------------------------------------------------------------------


def test_read_image_list_orders_and_finds_duplicates(tmp_path):
    listing = tmp_path / "list.txt"
    listing.write_text("a.jpg\n\nb.jpg\na.jpg\n", encoding="utf-8")
    result = read_image_list(listing)
    assert result.names == ["a.jpg", "b.jpg"]
    assert result.duplicates == ["a.jpg"]


def test_read_image_list_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_image_list(tmp_path / "does_not_exist.txt")


def test_list_image_files_ignores_non_images(clean_dataset):
    (clean_dataset["images_dir"] / "notes.txt").write_text("hello", encoding="utf-8")
    names = list_image_files(clean_dataset["images_dir"], (".jpg",))
    assert names == sorted(SYNTHETIC_IMAGE_IDS)
    assert "notes.txt" not in names


def test_find_missing_images(clean_dataset):
    (clean_dataset["images_dir"] / SYNTHETIC_IMAGE_IDS[0]).unlink()
    missing = find_missing_images(
        clean_dataset["images_dir"], SYNTHETIC_IMAGE_IDS, (".jpg",)
    )
    assert missing == [SYNTHETIC_IMAGE_IDS[0]]


def test_find_unreadable_images_detects_garbage_file(clean_dataset):
    broken = clean_dataset["images_dir"] / "broken.jpg"
    broken.write_bytes(b"this is definitely not a jpeg")
    unreadable = find_unreadable_images(
        clean_dataset["images_dir"], SYNTHETIC_IMAGE_IDS + ["broken.jpg"]
    )
    assert unreadable == ["broken.jpg"]


def test_find_unreadable_images_passes_valid_jpegs(clean_dataset):
    unreadable = find_unreadable_images(clean_dataset["images_dir"], SYNTHETIC_IMAGE_IDS)
    assert unreadable == []


# ---------------------------------------------------------------------------
# Unit tests: split analysis and assignment
# ---------------------------------------------------------------------------


def test_load_splits_clean(clean_dataset):
    splits = load_splits(
        clean_dataset["train_file"], clean_dataset["val_file"], clean_dataset["test_file"]
    )
    assert (len(splits.train), len(splits.validation), len(splits.test)) == (3, 1, 1)
    assert all(not ids for ids in splits.overlaps.values())
    assert all(not ids for ids in splits.duplicates.values())
    assert len(splits.all_split_ids()) == 5


def test_load_splits_detects_overlap_and_duplicates(clean_dataset):
    clean_dataset["train_file"].write_text(
        "\n".join(
            [SYNTHETIC_IMAGE_IDS[0], SYNTHETIC_IMAGE_IDS[0], SYNTHETIC_IMAGE_IDS[3]]
        )
        + "\n",
        encoding="utf-8",
    )
    splits = load_splits(
        clean_dataset["train_file"], clean_dataset["val_file"], clean_dataset["test_file"]
    )
    assert splits.duplicates["train"] == [SYNTHETIC_IMAGE_IDS[0]]
    assert splits.overlaps["train_and_validation"] == [SYNTHETIC_IMAGE_IDS[3]]


def test_load_splits_missing_file_raises(clean_dataset):
    with pytest.raises(FileNotFoundError):
        load_splits(
            clean_dataset["train_file"],
            clean_dataset["val_file"],
            clean_dataset["test_file"].parent / "absent.txt",
        )


def test_assign_splits_labels_each_image(clean_dataset):
    captions, splits = _prepare(clean_dataset)
    labeled = assign_splits(captions.captions, splits)
    assert set(labeled.columns) == {"image_id", "caption_id", "caption", "split"}
    assert set(labeled[labeled["image_id"] == SYNTHETIC_IMAGE_IDS[0]]["split"]) == {"train"}
    assert set(labeled[labeled["image_id"] == SYNTHETIC_IMAGE_IDS[3]]["split"]) == {"validation"}
    assert set(labeled[labeled["image_id"] == SYNTHETIC_IMAGE_IDS[4]]["split"]) == {"test"}


def test_assign_splits_marks_unassigned(clean_dataset):
    clean_dataset["train_file"].write_text(
        "\n".join(SYNTHETIC_IMAGE_IDS[1:3]) + "\n", encoding="utf-8"
    )
    captions, splits = _prepare(clean_dataset)
    labeled = assign_splits(captions.captions, splits)
    assert set(labeled[labeled["image_id"] == SYNTHETIC_IMAGE_IDS[0]]["split"]) == {
        "unassigned"
    }


def test_write_processed_csvs_counts_and_consistency(clean_dataset, tmp_path):
    captions, splits = _prepare(clean_dataset)
    labeled = assign_splits(captions.captions, splits)
    processed_dir = tmp_path / "processed"
    written = write_processed_csvs(labeled, processed_dir)

    assert set(written) == {"captions", "train", "validation", "test"}
    assert all(path.exists() for path in written.values())

    full = pd.read_csv(written["captions"])
    train = pd.read_csv(written["train"])
    validation = pd.read_csv(written["validation"])
    test = pd.read_csv(written["test"])

    assert list(full.columns) == ["image_id", "caption_id", "caption", "split"]
    assert (len(full), len(train), len(validation), len(test)) == (25, 15, 5, 5)
    assert set(train["image_id"]) & set(validation["image_id"]) == set()
    assert set(train["image_id"]) & set(test["image_id"]) == set()
    assert set(validation["image_id"]) & set(test["image_id"]) == set()

    resorted = full.sort_values(["image_id", "caption_id"]).reset_index(drop=True)
    assert full.equals(resorted)


# ---------------------------------------------------------------------------
# Unit tests: validation
# ---------------------------------------------------------------------------


def test_validate_clean_dataset_passes(clean_dataset):
    captions, splits = _prepare(clean_dataset)
    validation = validate_dataset(captions, splits, clean_dataset["images_dir"])
    assert validation.passed
    assert validation.error_count() == 0
    assert validation.warning_count() == 0


def test_validate_flags_missing_and_split_errors(clean_dataset):
    (clean_dataset["images_dir"] / SYNTHETIC_IMAGE_IDS[0]).unlink()
    clean_dataset["test_file"].write_text(
        SYNTHETIC_IMAGE_IDS[4] + "\nghost.jpg\n", encoding="utf-8"
    )
    captions, splits = _prepare(clean_dataset)
    validation = validate_dataset(captions, splits, clean_dataset["images_dir"])

    assert not validation.passed
    assert validation.missing_images == [SYNTHETIC_IMAGE_IDS[0]]
    assert "ghost.jpg" in validation.split_images_missing_on_disk
    assert "ghost.jpg" in validation.split_images_missing_from_captions


def test_validate_flags_corrupt_image_as_error(clean_dataset):
    corrupt_id = SYNTHETIC_IMAGE_IDS[2]
    (clean_dataset["images_dir"] / corrupt_id).write_bytes(b"garbage bytes")
    captions, splits = _prepare(clean_dataset)
    validation = validate_dataset(captions, splits, clean_dataset["images_dir"])
    assert not validation.passed
    assert validation.corrupt_images == [corrupt_id]


def test_validate_flags_empty_caption_as_warning(clean_dataset):
    lines = clean_dataset["captions_file"].read_text(encoding="utf-8").splitlines()
    lines[2] = f"{SYNTHETIC_IMAGE_IDS[0]}#2\t   "
    clean_dataset["captions_file"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    captions, splits = _prepare(clean_dataset)
    validation = validate_dataset(captions, splits, clean_dataset["images_dir"])
    assert validation.passed  # warning only, pipeline can continue
    assert validation.empty_caption_rows == [(SYNTHETIC_IMAGE_IDS[0], 2)]


def test_validate_flags_caption_count_violation_as_warning(clean_dataset):
    lines = [
        line
        for line in clean_dataset["captions_file"].read_text(encoding="utf-8").splitlines()
        if not line.startswith(f"{SYNTHETIC_IMAGE_IDS[0]}#4")
    ]
    clean_dataset["captions_file"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    captions, splits = _prepare(clean_dataset)
    validation = validate_dataset(captions, splits, clean_dataset["images_dir"])
    assert validation.passed
    assert validation.captions_per_image_violations == {SYNTHETIC_IMAGE_IDS[0]: 4}


def test_validate_reports_orphans_and_unassigned(clean_dataset):
    write_solid_jpeg(clean_dataset["images_dir"] / "extra_orphan.jpg")
    clean_dataset["train_file"].write_text(
        "\n".join(SYNTHETIC_IMAGE_IDS[1:3]) + "\n", encoding="utf-8"
    )
    captions, splits = _prepare(clean_dataset)
    validation = validate_dataset(captions, splits, clean_dataset["images_dir"])
    assert validation.passed  # warnings, not errors
    assert validation.orphan_image_files == ["extra_orphan.jpg"]
    assert validation.unassigned_images == [SYNTHETIC_IMAGE_IDS[0]]


def test_validate_skips_content_check_when_disabled(clean_dataset):
    (clean_dataset["images_dir"] / SYNTHETIC_IMAGE_IDS[1]).write_bytes(b"garbage")
    captions, splits = _prepare(clean_dataset)
    validation = validate_dataset(
        captions, splits, clean_dataset["images_dir"], verify_image_contents=False
    )
    assert validation.corrupt_images == []


# ---------------------------------------------------------------------------
# Unit tests: report
# ---------------------------------------------------------------------------


def test_report_statistics_match_synthetic_dataset(clean_dataset):
    captions, splits = _prepare(clean_dataset)
    labeled = assign_splits(captions.captions, splits)
    validation = validate_dataset(captions, splits, clean_dataset["images_dir"])
    stats = report.build_statistics(
        labeled, validation, clean_dataset["images_dir"]
    )
    assert stats["totals"] == {
        "images_on_disk": 5,
        "images_with_captions": 5,
        "total_captions": 25,
    }
    assert stats["captions_per_image"]["minimum"] == 5
    assert stats["captions_per_image"]["maximum"] == 5
    assert stats["captions_per_image"]["mean"] == 5.0
    assert stats["splits"]["train"] == {"images": 3, "captions": 15}
    assert stats["splits"]["validation"] == {"images": 1, "captions": 5}
    assert stats["splits"]["test"] == {"images": 1, "captions": 5}
    assert stats["splits"]["unassigned"] == {"images": 0, "captions": 0}
    assert stats["validation"]["passed"] is True
    assert stats["issues"]["missing_images"]["count"] == 0


def test_reports_written_and_json_round_trip(clean_dataset, tmp_path):
    captions, splits = _prepare(clean_dataset)
    labeled = assign_splits(captions.captions, splits)
    validation = validate_dataset(captions, splits, clean_dataset["images_dir"])
    stats = report.build_statistics(
        labeled, validation, clean_dataset["images_dir"]
    )

    json_path = report.write_json_report(stats, tmp_path / "reports" / "report.json")
    markdown_path = report.write_markdown_report(
        stats, tmp_path / "reports" / "report.md"
    )

    reloaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert reloaded["totals"] == stats["totals"]
    assert set(reloaded["issues"]) == set(stats["issues"])

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Flickr8k Dataset Validation Report" in markdown
    assert "PASSED" in markdown
    assert "missing_images" in markdown


# ---------------------------------------------------------------------------
# Unit tests: full pipeline entry point
# ---------------------------------------------------------------------------


def test_main_pipeline_on_synthetic_dataset(clean_dataset, tmp_path, capsys):
    config_path, processed_dir = _write_config(clean_dataset, tmp_path)
    exit_code = phase1_main(["--config", str(config_path)])
    assert exit_code == 0

    assert (processed_dir / "captions.csv").exists()
    assert (processed_dir / "train.csv").exists()
    assert (processed_dir / "validation.csv").exists()
    assert (processed_dir / "test.csv").exists()

    stats = json.loads(
        (tmp_path / "reports" / "dataset_report.json").read_text(encoding="utf-8")
    )
    assert stats["totals"]["total_captions"] == 25
    assert stats["validation"]["passed"] is True

    output = capsys.readouterr().out
    assert "PASSED" in output
    assert "Total captions" in output


def test_main_skips_csvs_when_validation_fails(clean_dataset, tmp_path, capsys):
    config_path, processed_dir = _write_config(clean_dataset, tmp_path)
    (clean_dataset["images_dir"] / SYNTHETIC_IMAGE_IDS[0]).unlink()
    exit_code = phase1_main(["--config", str(config_path)])
    assert exit_code == 1
    assert not (processed_dir / "captions.csv").exists()
    assert (tmp_path / "reports" / "dataset_report.json").exists()
    assert "SKIPPED" in capsys.readouterr().out


def test_main_returns_error_when_dataset_missing(tmp_path, capsys):
    config = {
        "paths": {
            "images_dir": str(tmp_path / "nowhere" / "Images"),
            "captions_file": str(tmp_path / "nowhere" / "Flickr8k.token.txt"),
            "train_split_file": str(tmp_path / "nowhere" / "train.txt"),
            "val_split_file": str(tmp_path / "nowhere" / "val.txt"),
            "test_split_file": str(tmp_path / "nowhere" / "test.txt"),
            "processed_dir": str(tmp_path / "processed"),
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    exit_code = phase1_main(["--config", str(config_path)])
    assert exit_code == 2
    assert "missing" in capsys.readouterr().err.lower()


# ---------------------------------------------------------------------------
# Unit tests: project-level exclusions (dataset.excluded_images)
# ---------------------------------------------------------------------------


def test_exclusion_entries_require_id_and_reason():
    assert parse_exclusions(None) == []
    assert parse_exclusions([]) == []
    with pytest.raises(ValueError):
        parse_exclusions([{"id": "x.jpg"}])  # missing reason
    with pytest.raises(ValueError):
        parse_exclusions([{"id": "", "reason": "y"}])  # empty id
    with pytest.raises(ValueError):
        parse_exclusions([{"id": "a.jpg", "reason": "r"}, {"id": "a.jpg", "reason": "s"}])
    entries = parse_exclusions([{"id": "a.jpg", "reason": "r"}])
    assert entries[0].image_id == "a.jpg"
    assert entries[0].reason == "r"


def test_apply_exclusions_removes_exactly_target_rows(clean_dataset):
    lines = clean_dataset["captions_file"].read_text(encoding="utf-8").rstrip("\n").split("\n")
    phantom_lines = [f"phantom.jpg.1#{index}\tphantom caption {index}" for index in range(5)]
    clean_dataset["captions_file"].write_text(
        "\n".join(lines + phantom_lines) + "\n", encoding="utf-8"
    )
    captions = load_caption_file(clean_dataset["captions_file"])
    result = apply_exclusions(
        captions, parse_exclusions([{"id": "phantom.jpg.1", "reason": "official defect"}])
    )

    assert len(result.remaining.captions) == 25
    assert "phantom.jpg.1" not in set(result.remaining.captions["image_id"])
    assert len(result.removed) == 5
    assert result.removed.iloc[0]["caption"] == "phantom caption 0"
    assert result.removed_ids == ["phantom.jpg.1"]
    assert result.unknown_ids == []

    original = captions.captions[captions.captions["image_id"] != "phantom.jpg.1"]
    assert result.remaining.captions.equals(original.reset_index(drop=True))


def test_apply_exclusions_reports_unknown_ids(clean_dataset):
    captions, _ = _prepare(clean_dataset)
    result = apply_exclusions(
        captions, parse_exclusions([{"id": "ghost.jpg", "reason": "typo"}])
    )
    assert result.unknown_ids == ["ghost.jpg"]
    assert result.removed_ids == []
    assert len(result.remaining.captions) == 25  # untouched


def test_pipeline_still_fails_without_exclusion(clean_dataset, tmp_path):
    """The validator stays strict: an unexcluded phantom ID must fail the run."""
    config_path, _ = _write_config(clean_dataset, tmp_path)
    with open(clean_dataset["captions_file"], "a", encoding="utf-8") as handle:
        for index in range(5):
            handle.write(f"phantom.jpg.1#{index}\tphantom caption {index}\n")
    assert phase1_main(["--config", str(config_path)]) == 1


def test_pipeline_applies_configured_exclusion(clean_dataset, tmp_path):
    config_path, processed_dir = _write_config(clean_dataset, tmp_path)
    with open(clean_dataset["captions_file"], "a", encoding="utf-8") as handle:
        for index in range(5):
            handle.write(f"phantom.jpg.1#{index}\tphantom caption {index}\n")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["dataset"]["excluded_images"] = [
        {"id": "phantom.jpg.1", "reason": "synthetic upstream defect"}
    ]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert phase1_main(["--config", str(config_path)]) == 0

    full = pd.read_csv(processed_dir / "captions.csv")
    assert "phantom.jpg.1" not in set(full["image_id"])
    assert len(full) == 25
    assert (full.groupby("image_id").size() == 5).all()

    stats = json.loads(
        (tmp_path / "reports" / "dataset_report.json").read_text(encoding="utf-8")
    )
    assert stats["original_source"] == {"referenced_image_ids": 6, "captions": 30}
    assert stats["totals"]["total_captions"] == 25
    assert stats["exclusions"]["removed_captions_total"] == 5
    assert stats["exclusions"]["entries"][0]["image_id"] == "phantom.jpg.1"
    assert stats["exclusions"]["stale_ids"] == []

    markdown = (tmp_path / "reports" / "dataset_report.md").read_text(encoding="utf-8")
    assert "Original Flickr8k source" in markdown
    assert "Intentional project exclusions" in markdown
    assert "not the complete original Flickr8k dataset" in markdown


# ---------------------------------------------------------------------------
# Integration tests: real Flickr8k (skipped until the dataset is present)
# ---------------------------------------------------------------------------


@requires_real_dataset
class TestRealFlickr8k:
    """End-to-end checks against the real archive (skipped if absent)."""

    def test_raw_files_exist(self):
        for name in (
            "Flickr8k.token.txt",
            "Flickr_8k.trainImages.txt",
            "Flickr_8k.devImages.txt",
            "Flickr_8k.testImages.txt",
        ):
            assert (RAW_DIR / name).exists(), name
        assert (RAW_DIR / "Images").is_dir()

    def test_captions_reference_existing_images(self):
        captions = load_caption_file(RAW_DIR / "Flickr8k.token.txt")
        assert len(captions.captions) > 0
        raw_missing = find_missing_images(
            RAW_DIR / "Images", set(captions.captions["image_id"]), (".jpg",)
        )
        # The official release references exactly one image ID that has no
        # file in the official image archive (known upstream defect —
        # docs/phase1_dataset.md, sections 12-13). The project excludes it
        # via dataset.excluded_images, so the *project* caption set must
        # reference existing images only.
        config = load_config()
        excluded_ids = {
            entry["id"] for entry in config["dataset"].get("excluded_images", [])
        }
        assert set(raw_missing) == excluded_ids == {"2258277193_586949ec62.jpg.1"}
        project = apply_exclusions(
            captions,
            parse_exclusions(config["dataset"].get("excluded_images")),
        )
        assert find_missing_images(
            RAW_DIR / "Images", set(project.remaining.captions["image_id"]), (".jpg",)
        ) == []

    def test_images_readable(self):
        captions = load_caption_file(RAW_DIR / "Flickr8k.token.txt")
        unreadable = find_unreadable_images(
            RAW_DIR / "Images", set(captions.captions["image_id"])
        )
        assert unreadable == []

    def test_splits_disjoint_and_complete(self):
        splits = load_splits(
            RAW_DIR / "Flickr_8k.trainImages.txt",
            RAW_DIR / "Flickr_8k.devImages.txt",
            RAW_DIR / "Flickr_8k.testImages.txt",
        )
        for split in ("train", "validation", "test"):
            assert len(splits.names_for(split)) > 0
            assert splits.duplicates[split] == []
        assert all(not ids for ids in splits.overlaps.values())

    def test_pipeline_end_to_end(self):
        assert phase1_main([]) == 0

        processed = PROJECT_ROOT / "data" / "processed"
        full = pd.read_csv(processed / "captions.csv")
        train = pd.read_csv(processed / "train.csv")
        validation = pd.read_csv(processed / "validation.csv")
        test = pd.read_csv(processed / "test.csv")

        assert list(full.columns) == ["image_id", "caption_id", "caption", "split"]
        # the documented exclusion is applied and every remaining image has
        # exactly five captions
        assert "2258277193_586949ec62.jpg.1" not in set(full["image_id"])
        assert (full.groupby("image_id").size() == 5).all()
        assert len(full) == len(train) + len(validation) + len(test) + len(
            full[full["split"] == "unassigned"]
        )
        assert set(train["image_id"]) & set(validation["image_id"]) == set()
        assert set(train["image_id"]) & set(test["image_id"]) == set()
        assert set(validation["image_id"]) & set(test["image_id"]) == set()
