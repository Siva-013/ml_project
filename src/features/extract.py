"""Phase 3 — feature extraction interface and dataset-level extraction.

``extract_features`` maps one Phase 2 :class:`PreprocessedImage` to the three
individual classical feature vectors (no fusion — concatenation belongs to
Phase 4).

``extract_dataset`` runs the extraction over the whole project dataset in a
single image-loading pass and stores the three feature matrices together
with their aligned image IDs and split labels:

    data/features/
    ├── image_ids.npy          (N,)     strings
    ├── splits.npy             (N,)     strings: train | validation | test
    ├── hog_features.npy       (N, D_hog)   float32
    ├── color_features.npy     (N, 96)      float32
    ├── lbp_features.npy       (N, 10)      float32
    └── feature_metadata.json

Feature extraction is deterministic and fits nothing (no PCA, no learned
parameters), so no train/validation/test separation is needed at this stage;
the split labels are stored so later phases can honour the data-leakage
rules.

Run from the project root::

    python -m src.features.extract [--splits train validation test] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import load_config, resolve_path
from src.features.color_histogram import (
    COLOR_BINS_PER_CHANNEL,
    COLOR_NORMALIZATION,
    COLOR_SPACE,
    extract_color_histogram,
)
from src.features.hog import (
    HOG_BLOCK_NORM,
    HOG_CELLS_PER_BLOCK,
    HOG_ORIENTATIONS,
    HOG_PIXELS_PER_CELL,
    HOG_TRANSFORM_SQRT,
    extract_hog,
)
from src.features.lbp import LBP_BINS, LBP_METHOD, LBP_POINTS, LBP_RADIUS, extract_lbp
from src.preprocessing.image_preprocessing import (
    TARGET_SIZE,
    PreprocessedImage,
    preprocess_image,
)

SPLIT_NAMES: tuple[str, ...] = ("train", "validation", "test")

#: Splits the extractor covers by default. ``unassigned`` images are part of
#: the project dataset; extraction is deterministic and fits nothing, so
#: computing their features breaks no data-leakage rule, and their stored
#: label lets later phases exclude them from any fitting.
EXTRACTION_SPLITS: tuple[str, ...] = SPLIT_NAMES + ("unassigned",)

#: Names of the saved feature files, in stable order.
FEATURE_FILE_NAMES = (
    "image_ids.npy",
    "splits.npy",
    "hog_features.npy",
    "color_features.npy",
    "lbp_features.npy",
)


def extract_features(image: PreprocessedImage) -> dict[str, np.ndarray]:
    """Extract all three classical feature vectors for one preprocessed image.

    Args:
        image: The Phase 2 preprocessing output.

    Returns:
        A dict with the individual, un-fused feature vectors:

        * ``"hog"``   — 1-D float64 vector (shape, texture, contours)
        * ``"color"`` — 1-D float64 vector of length 96 (colour distribution)
        * ``"lbp"``   — 1-D float64 vector of length 10 (texture)
    """
    return {
        "hog": extract_hog(image.grayscale),
        "color": extract_color_histogram(image.hsv),
        "lbp": extract_lbp(image.grayscale),
    }


def load_split_assignments(
    processed_dir: Path, splits: Sequence[str] = SPLIT_NAMES
) -> list[tuple[str, str]]:
    """Return ``(image_id, split)`` pairs for the requested splits.

    The mapping comes from ``captions.csv`` (Phase 1 output), where every
    image carries exactly one split label. Order follows the CSV, which is
    sorted by ``(image_id, caption_id)`` — deterministic across runs.
    """
    captions_path = processed_dir / "captions.csv"
    if not captions_path.exists():
        raise FileNotFoundError(
            f"Processed captions not found at {captions_path}. "
            "Run 'python -m src.dataset' first."
        )
    captions = pd.read_csv(captions_path)
    if not {"image_id", "split"}.issubset(captions.columns):
        raise ValueError(f"{captions_path} lacks the required columns image_id/split")

    assignments = captions[["image_id", "split"]].drop_duplicates(
        subset="image_id", keep="first"
    )
    selected = assignments[assignments["split"].isin(splits)]
    if selected.empty:
        raise ValueError(f"No images found for splits {list(splits)}")
    return [
        (str(row.image_id), str(row.split)) for row in selected.itertuples(index=False)
    ]


def extract_dataset(
    images_dir: Path,
    processed_dir: Path,
    output_dir: Path,
    *,
    splits: Sequence[str] = SPLIT_NAMES,
    limit: int | None = None,
    show_progress: bool = True,
) -> dict:
    """Extract features for the selected splits and save aligned matrices.

    Every image is loaded exactly once; all three features are computed in
    the same pass. A failure on any image aborts the run with the offending
    image ID — images are never silently skipped (Phase 1 has already
    validated the dataset, so a failure here is a data-integrity problem).

    Returns:
        The metadata dictionary that is also written to
        ``feature_metadata.json``.
    """
    assignments = load_split_assignments(processed_dir, splits)
    if limit is not None:
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        assignments = assignments[:limit]

    image_ids = [image_id for image_id, _ in assignments]
    split_labels = [split for _, split in assignments]

    hog_rows: list[np.ndarray] = []
    color_rows: list[np.ndarray] = []
    lbp_rows: list[np.ndarray] = []
    start = time.perf_counter()
    progress = tqdm(
        assignments,
        desc="Extracting features (HOG + Color + LBP)",
        disable=not show_progress,
        unit="img",
    )
    for image_id, _split in progress:
        try:
            preprocessed = preprocess_image(images_dir / image_id, image_id=image_id)
            features = extract_features(preprocessed)
        except Exception as error:
            raise RuntimeError(
                f"Feature extraction failed for image '{image_id}': {error}"
            ) from error
        hog_rows.append(features["hog"])
        color_rows.append(features["color"])
        lbp_rows.append(features["lbp"])
    elapsed = time.perf_counter() - start

    hog_matrix = np.stack(hog_rows).astype(np.float32)
    color_matrix = np.stack(color_rows).astype(np.float32)
    lbp_matrix = np.stack(lbp_rows).astype(np.float32)
    matrices = {"hog": hog_matrix, "color": color_matrix, "lbp": lbp_matrix}
    for name, matrix in matrices.items():
        if not np.isfinite(matrix).all():
            raise RuntimeError(f"Non-finite values found in {name} feature matrix")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "image_ids.npy", np.asarray(image_ids, dtype="U64"))
    np.save(output_dir / "splits.npy", np.asarray(split_labels, dtype="U16"))
    np.save(output_dir / "hog_features.npy", hog_matrix)
    np.save(output_dir / "color_features.npy", color_matrix)
    np.save(output_dir / "lbp_features.npy", lbp_matrix)

    split_counts = Counter(split_labels)
    metadata = {
        "dataset": "Flickr8k (project dataset; 1 documented upstream exclusion)",
        "project_images": int(len(image_ids)),
        "split_counts": {name: int(split_counts.get(name, 0)) for name in splits},
        "feature_types": ["HOG", "Color Histogram", "LBP"],
        "preprocessing": {
            "target_size": [TARGET_SIZE[0], TARGET_SIZE[1]],
            "resize_filter": "Lanczos",
            "grayscale": "Rec. 709 luma (skimage.color.rgb2gray)",
            "value_range": "[0, 1] float32",
        },
        "hog": {
            "orientations": HOG_ORIENTATIONS,
            "pixels_per_cell": list(HOG_PIXELS_PER_CELL),
            "cells_per_block": list(HOG_CELLS_PER_BLOCK),
            "block_norm": HOG_BLOCK_NORM,
            "transform_sqrt": HOG_TRANSFORM_SQRT,
            "dimension": int(hog_matrix.shape[1]),
        },
        "color_histogram": {
            "color_space": COLOR_SPACE,
            "bins_per_channel": COLOR_BINS_PER_CHANNEL,
            "channel_order": ["H", "S", "V"],
            "normalization": COLOR_NORMALIZATION,
            "dimension": int(color_matrix.shape[1]),
        },
        "lbp": {
            "points": LBP_POINTS,
            "radius": LBP_RADIUS,
            "method": LBP_METHOD,
            "bins": LBP_BINS,
            "dimension": int(lbp_matrix.shape[1]),
        },
        "storage": {"dtype": "float32", "files": list(FEATURE_FILE_NAMES)},
        "extraction": {
            "images_processed": int(len(image_ids)),
            "failed": 0,
            "elapsed_seconds": round(elapsed, 2),
            "splits": list(splits),
        },
    }
    (output_dir / "feature_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.features.extract",
        description="Phase 3: extract HOG, colour histogram and LBP features "
        "for the project dataset.",
    )
    parser.add_argument(
        "--config", default=None, help="Path to config.yaml (default: project root)"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(EXTRACTION_SPLITS),
        choices=EXTRACTION_SPLITS,
        help="Dataset splits to process (default: train, validation, test, unassigned)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N images (debugging)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Feature output directory (default: paths.features_dir from config)",
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable the tqdm progress bar"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        metadata = extract_dataset(
            images_dir=resolve_path(config["paths"]["images_dir"]),
            processed_dir=resolve_path(config["paths"]["processed_dir"]),
            output_dir=(
                resolve_path(args.output_dir)
                if args.output_dir
                else resolve_path(config["paths"]["features_dir"])
            ),
            splits=args.splits,
            limit=args.limit,
            show_progress=not args.no_progress,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"[error] {error}", file=sys.stderr)
        return 1

    extraction = metadata["extraction"]
    print("Phase 3 feature extraction complete:")
    print(
        f"  images processed : {extraction['images_processed']} "
        f"({extraction['failed']} failed) in {extraction['elapsed_seconds']} s"
    )
    print(f"  split counts     : {metadata['split_counts']}")
    print(
        f"  dimensions       : HOG={metadata['hog']['dimension']}, "
        f"Color={metadata['color_histogram']['dimension']}, "
        f"LBP={metadata['lbp']['dimension']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
