"""Phase 4 — feature fusion and PCA, fitted on training data only.

Combines the three Phase 3 handcrafted feature families into one compact
visual representation:

    HOG (8100) ─┐
    Color (96) ─┼─ per-family StandardScaler (FIT: train only, z = (x-μ)/σ)
    LBP (10)  ──┘         ↓
                fusion (concatenation) = 8,206 dimensions
                          ↓
                PCA, 95 % variance target (FIT: train only)
                          ↓
    compact per-split matrices + saved scalers + saved PCA

Data-leakage rules enforced by construction: the ``fit_*`` functions only
ever receive the training rows (their signatures take train arrays), and
every other split is passed exclusively through ``transform_and_fuse`` /
``PCA.transform`` using those training-fitted objects. The 91 unassigned
images are transformed as well but never influence any fit, and are stored
separately from train/validation/test.

Outputs (Phase 3 files are read, never modified)::

    data/features/fusion/
    ├── train_features.npy            (6000, D_pca) float32
    ├── validation_features.npy       (1000, D_pca)
    ├── test_features.npy             (1000, D_pca)
    ├── unassigned_features.npy       (  91, D_pca)
    ├── train_image_ids.npy / validation_image_ids.npy /
    │   test_image_ids.npy / unassigned_image_ids.npy
    ├── pca.joblib                    the fitted PCA (reusable at inference)
    ├── scalers/
    │   ├── hog_scaler.joblib
    │   ├── color_scaler.joblib
    │   └── lbp_scaler.joblib
    └── fusion_metadata.json

Run from the project root::

    python -m src.features.fusion
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import load_config, resolve_path

#: Sub-directory of ``paths.features_dir`` holding all Phase 4 outputs.
FUSION_SUBDIR = "fusion"

#: Explained-variance target used to select the number of PCA components.
TARGET_EXPLAINED_VARIANCE = 0.95

#: Recorded for completeness; the full SVD solver is deterministic, so no
#: randomness is involved anywhere in Phase 4.
PCA_RANDOM_STATE = 42

#: Fusion order of the three Phase 3 feature families.
FAMILY_ORDER: tuple[str, ...] = ("hog", "color_histogram", "lbp")

#: Per-family input dimensions produced by Phase 3.
EXPECTED_INPUT_DIMENSIONS: dict[str, int] = {
    "hog": 8100,
    "color_histogram": 96,
    "lbp": 10,
}

#: Dimension of the fused representation before PCA: 8100 + 96 + 10.
FUSED_DIMENSION = sum(EXPECTED_INPUT_DIMENSIONS.values())

#: Split labels present in the Phase 3 ``splits.npy``.
SPLIT_LABELS: tuple[str, ...] = ("train", "validation", "test", "unassigned")

_PHASE3_FILES = (
    "image_ids.npy",
    "splits.npy",
    "hog_features.npy",
    "color_features.npy",
    "lbp_features.npy",
)


@dataclass(frozen=True)
class Phase3Features:
    """The five aligned Phase 3 arrays."""

    image_ids: np.ndarray
    splits: np.ndarray
    matrices: dict[str, np.ndarray]


@dataclass
class FittedTransform:
    """The transformations fitted on the training rows only.

    Attributes:
        scalers: One :class:`StandardScaler` per feature family.
        pca: The fitted PCA model.
        train_fused: The standardized, fused **training** matrix that the
            PCA was fitted on (kept for reference; not persisted).
    """

    scalers: dict[str, StandardScaler]
    pca: PCA
    train_fused: np.ndarray


def load_phase3_features(features_dir: Path) -> Phase3Features:
    """Load and shape-check the five Phase 3 arrays.

    Raises:
        FileNotFoundError: If any Phase 3 file is missing.
        ValueError: If row counts or per-family dimensions are inconsistent.
    """
    features_dir = Path(features_dir)
    missing = [name for name in _PHASE3_FILES if not (features_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing Phase 3 files under {features_dir}: {missing}. "
            "Run 'python -m src.features.extract' first."
        )
    image_ids = np.load(features_dir / "image_ids.npy")
    splits = np.load(features_dir / "splits.npy")
    matrices = {
        "hog": np.load(features_dir / "hog_features.npy"),
        "color_histogram": np.load(features_dir / "color_features.npy"),
        "lbp": np.load(features_dir / "lbp_features.npy"),
    }

    row_counts = {len(image_ids), len(splits)} | {len(m) for m in matrices.values()}
    if len(row_counts) != 1:
        raise ValueError(f"Phase 3 arrays have misaligned row counts: {row_counts}")
    for family, matrix in matrices.items():
        if matrix.shape[1] != EXPECTED_INPUT_DIMENSIONS[family]:
            raise ValueError(
                f"{family} features have {matrix.shape[1]} columns, expected "
                f"{EXPECTED_INPUT_DIMENSIONS[family]}"
            )
    return Phase3Features(
        image_ids=image_ids, splits=splits, matrices=matrices
    )


def split_masks(splits: np.ndarray) -> dict[str, np.ndarray]:
    """Return a boolean mask into the dataset for every split label.

    Raises:
        ValueError: If a split label is unknown or a split is empty.
    """
    masks: dict[str, np.ndarray] = {}
    for label in SPLIT_LABELS:
        masks[label] = splits == label
    unknown = sorted(set(splits.tolist()) - set(SPLIT_LABELS))
    if unknown:
        raise ValueError(f"Unknown split labels in splits.npy: {unknown}")
    for label, mask in masks.items():
        if not mask.any():
            raise ValueError(f"Split '{label}' contains no rows")
    return masks


def fuse_features(scaled: Mapping[str, np.ndarray]) -> np.ndarray:
    """Concatenate the standardized families (FAMILY_ORDER) along axis 1.

    Raises:
        ValueError: If a family is missing or the fused width is not
            :data:`FUSED_DIMENSION`.
    """
    if set(scaled) != set(FAMILY_ORDER):
        raise ValueError(
            f"Expected exactly the families {FAMILY_ORDER}, got {sorted(scaled)}"
        )
    fused = np.concatenate([scaled[family] for family in FAMILY_ORDER], axis=1)
    if fused.shape[1] != FUSED_DIMENSION:
        raise ValueError(
            f"Fused matrix has {fused.shape[1]} columns, expected {FUSED_DIMENSION}"
        )
    return fused


def fit_scalers(
    train_matrices: Mapping[str, np.ndarray]
) -> dict[str, StandardScaler]:
    """FIT step: one StandardScaler per family, on training rows only.

    Each family gets its own scaler so that its z-scores are computed from
    its own training statistics — the three families have very different
    scales and distributions.

    Raises:
        ValueError: If a training matrix is empty or contains non-finite
            values.
    """
    scalers: dict[str, StandardScaler] = {}
    for family in FAMILY_ORDER:
        matrix = train_matrices[family]
        if len(matrix) == 0:
            raise ValueError(f"Cannot fit the {family} scaler on zero rows")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{family} training features contain non-finite values")
        scalers[family] = StandardScaler().fit(matrix)
    return scalers


def transform_and_fuse(
    scalers: Mapping[str, StandardScaler],
    matrices: Mapping[str, np.ndarray],
) -> np.ndarray:
    """TRANSFORM step: apply the training-fitted scalers, then fuse.

    Raises:
        ValueError: If a transformed family contains non-finite values.
    """
    scaled = {
        family: scalers[family].transform(matrices[family])
        for family in FAMILY_ORDER
    }
    fused = fuse_features(scaled)
    if not np.isfinite(fused).all():
        raise ValueError("Standardized/fused features contain non-finite values")
    return fused


def fit_pca(
    train_fused: np.ndarray,
    target_variance: float = TARGET_EXPLAINED_VARIANCE,
) -> PCA:
    """FIT step: PCA on the standardized, fused **training** matrix only.

    ``n_components`` is given as a variance fraction, so scikit-learn
    selects the smallest component count reaching the target from the real
    training data — the number is never hard-coded. The exact ``full`` SVD
    solver is deterministic (no randomness), so the pipeline is
    reproducible bit-for-bit on the same inputs.

    Raises:
        ValueError: If the training matrix is empty or non-finite.
    """
    if len(train_fused) == 0:
        raise ValueError("Cannot fit PCA on zero rows")
    if not np.isfinite(train_fused).all():
        raise ValueError("Fused training features contain non-finite values")
    pca = PCA(
        n_components=target_variance,
        svd_solver="full",
        random_state=PCA_RANDOM_STATE,
    )
    pca.fit(train_fused)
    return pca


def run_fusion(
    features_dir: Path,
    output_dir: Path,
    *,
    target_variance: float = TARGET_EXPLAINED_VARIANCE,
) -> dict:
    """Run the complete Phase 4 pipeline and persist every artifact.

    Returns:
        The metadata dictionary that is also written to
        ``fusion_metadata.json``.
    """
    features = load_phase3_features(features_dir)
    masks = split_masks(features.splits)
    output_dir = Path(output_dir)
    (output_dir / "scalers").mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    # ---- FIT (training rows only) -------------------------------------
    train_matrices = {
        family: features.matrices[family][masks["train"]]
        for family in FAMILY_ORDER
    }
    scalers = fit_scalers(train_matrices)
    train_fused = transform_and_fuse(scalers, train_matrices)
    pca = fit_pca(train_fused, target_variance)

    # ---- TRANSFORM (every split, with the training-fitted objects) ----
    split_counts: dict[str, int] = {}
    for label in SPLIT_LABELS:
        rows = {
            family: features.matrices[family][masks[label]]
            for family in FAMILY_ORDER
        }
        fused = transform_and_fuse(scalers, rows)
        reduced = pca.transform(fused).astype(np.float32, copy=False)
        if not np.isfinite(reduced).all():
            raise RuntimeError(f"PCA features for split '{label}' are not finite")
        np.save(output_dir / f"{label}_features.npy", reduced)
        np.save(output_dir / f"{label}_image_ids.npy", features.image_ids[masks[label]])
        split_counts[label] = int(masks[label].sum())
        del rows, fused, reduced

    # ---- Persist the fitted transformations ---------------------------
    scaler_files = {
        "hog": "hog_scaler.joblib",
        "color_histogram": "color_scaler.joblib",
        "lbp": "lbp_scaler.joblib",
    }
    for family, filename in scaler_files.items():
        joblib.dump(scalers[family], output_dir / "scalers" / filename)
    joblib.dump(pca, output_dir / "pca.joblib")

    elapsed = time.perf_counter() - start
    metadata = {
        "dataset": "Flickr8k (project dataset; 1 documented upstream exclusion)",
        "project_images": int(len(features.image_ids)),
        "input_dimensions": {
            "hog": EXPECTED_INPUT_DIMENSIONS["hog"],
            "color_histogram": EXPECTED_INPUT_DIMENSIONS["color_histogram"],
            "lbp": EXPECTED_INPUT_DIMENSIONS["lbp"],
            "fused": FUSED_DIMENSION,
        },
        "split_counts": split_counts,
        "standardization": {
            "method": "StandardScaler",
            "fit_on": "train_only",
            "formula": "z = (x - mean) / std, using training-set statistics",
            "train_samples_seen": {
                family: int(scalers[family].n_samples_seen_)
                for family in FAMILY_ORDER
            },
        },
        "pca": {
            "method": "PCA",
            "fit_on": "train_only",
            "target_explained_variance": float(target_variance),
            "svd_solver": "full",
            "components": int(pca.n_components_),
            "reduced_dimension": int(pca.n_components_),
            "explained_variance": float(pca.explained_variance_ratio_.sum()),
            "train_samples": int(masks["train"].sum()),
        },
        "dtype": "float32",
        "outputs": {
            "feature_files": [
                f"{label}_features.npy" for label in SPLIT_LABELS
            ],
            "id_files": [f"{label}_image_ids.npy" for label in SPLIT_LABELS],
            "pca": "pca.joblib",
            "scalers": [f"scalers/{file}" for file in scaler_files.values()],
            "metadata": "fusion_metadata.json",
        },
        "runtime": {"elapsed_seconds": round(elapsed, 2)},
    }
    (output_dir / "fusion_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.features.fusion",
        description="Phase 4: standardize (train-only), fuse (8206-D) and "
        "reduce with PCA (train-only, 95 % variance).",
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--features-dir",
        default=None,
        help="Phase 3 feature directory (default: paths.features_dir)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Fusion output directory (default: <features_dir>/fusion)",
    )
    parser.add_argument(
        "--target-variance",
        type=float,
        default=TARGET_EXPLAINED_VARIANCE,
        help=f"PCA explained-variance target (default: {TARGET_EXPLAINED_VARIANCE})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        features_dir = (
            resolve_path(args.features_dir)
            if args.features_dir
            else resolve_path(config["paths"]["features_dir"])
        )
        output_dir = (
            resolve_path(args.output_dir)
            if args.output_dir
            else features_dir / FUSION_SUBDIR
        )
        metadata = run_fusion(
            features_dir, output_dir, target_variance=args.target_variance
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"[error] {error}", file=sys.stderr)
        return 1

    pca_info = metadata["pca"]
    print("Phase 4 fusion + PCA complete:")
    print(f"  images           : {metadata['project_images']} {metadata['split_counts']}")
    print(
        f"  input dimensions : {metadata['input_dimensions']} "
        f"(fused = {metadata['input_dimensions']['fused']})"
    )
    print(
        f"  PCA              : {pca_info['components']} components "
        f"({pca_info['explained_variance']:.4f} explained variance, "
        f"fitted on {pca_info['train_samples']} training rows)"
    )
    print(f"  outputs          : {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
