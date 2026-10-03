"""Phase 4 tests: standardization, fusion and PCA.

Unit tests run the full fusion pipeline on small synthetic matrices that
use the real Phase 3 dimensions (8100 / 96 / 10), so FIT/TRANSFORM
separation, the train-only leakage rule, dimensions, determinism and
alignment are all exercised quickly. Integration tests verify the real
artifacts under ``data/features/fusion/`` once the pipeline has been run,
and confirm the Phase 3 inputs remain untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.decomposition import PCA

from src.features.fusion import (
    EXPECTED_INPUT_DIMENSIONS,
    FUSED_DIMENSION,
    SPLIT_LABELS,
    TARGET_EXPLAINED_VARIANCE,
    fit_pca,
    fit_scalers,
    fuse_features,
    load_phase3_features,
    run_fusion,
    split_masks,
    transform_and_fuse,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
FUSION_DIR = FEATURES_DIR / "fusion"
PHASE3_AVAILABLE = (FEATURES_DIR / "hog_features.npy").exists()
FUSION_AVAILABLE = (FUSION_DIR / "pca.joblib").exists()


def _synthetic_dataset(seed: int = 7):
    """Small synthetic dataset with the real Phase 3 family dimensions.

    Rows after the training block are shifted by +25 in HOG space, so any
    leakage of non-training rows into a fitted scaler is detectable: a
    correct scaler's means must match the training means, not the shifted
    global means.
    """
    rng = np.random.default_rng(seed)
    counts = {"train": 20, "validation": 6, "test": 6, "unassigned": 4}
    total = sum(counts.values())
    hog = rng.normal(0.0, 1.0, (total, EXPECTED_INPUT_DIMENSIONS["hog"])).astype(np.float32)
    color = (rng.random((total, EXPECTED_INPUT_DIMENSIONS["color_histogram"])) * 10).astype(np.float32)
    lbp = (rng.random((total, EXPECTED_INPUT_DIMENSIONS["lbp"])) * 5).astype(np.float32)
    shift = np.zeros(total)
    shift[counts["train"]:] = 25.0
    hog = hog + shift[:, None]
    splits = np.array(
        ["train"] * counts["train"]
        + ["validation"] * counts["validation"]
        + ["test"] * counts["test"]
        + ["unassigned"] * counts["unassigned"]
    )
    image_ids = np.array([f"img_{index:03d}.jpg" for index in range(total)], dtype="U64")
    matrices = {"hog": hog, "color_histogram": color, "lbp": lbp}
    return counts, splits, image_ids, matrices


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------


def test_input_dimensions_match_phase3():
    assert EXPECTED_INPUT_DIMENSIONS == {
        "hog": 8100,
        "color_histogram": 96,
        "lbp": 10,
    }


def test_fused_dimension_is_8206():
    _counts, splits, _ids, matrices = _synthetic_dataset()
    masks = split_masks(splits)
    train = {family: matrices[family][masks["train"]] for family in matrices}
    scalers = fit_scalers(train)
    fused = transform_and_fuse(scalers, train)
    assert fused.shape[1] == 8100 + 96 + 10
    assert fused.shape[1] == FUSED_DIMENSION == 8206


def test_fuse_features_rejects_missing_family():
    with pytest.raises(ValueError, match="families"):
        fuse_features({"hog": np.zeros((2, 10))})


# ---------------------------------------------------------------------------
# Split shapes and standardization quality
# ---------------------------------------------------------------------------


def test_per_split_fused_shapes():
    counts, splits, _ids, matrices = _synthetic_dataset()
    masks = split_masks(splits)
    train = {family: matrices[family][masks["train"]] for family in matrices}
    scalers = fit_scalers(train)
    for label in SPLIT_LABELS:
        rows = {family: matrices[family][masks[label]] for family in matrices}
        fused = transform_and_fuse(scalers, rows)
        assert fused.shape == (counts[label], FUSED_DIMENSION)


def test_training_standardization_mean_zero_std_one():
    _counts, splits, _ids, matrices = _synthetic_dataset()
    masks = split_masks(splits)
    train = {family: matrices[family][masks["train"]] for family in matrices}
    scalers = fit_scalers(train)
    for family in ("hog", "color_histogram", "lbp"):
        scaled = scalers[family].transform(train[family])
        assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-4), family
        assert np.allclose(scaled.std(axis=0), 1.0, atol=1e-3), family


def test_every_output_is_finite():
    _counts, splits, _ids, matrices = _synthetic_dataset()
    masks = split_masks(splits)
    train = {family: matrices[family][masks["train"]] for family in matrices}
    scalers = fit_scalers(train)
    train_fused = transform_and_fuse(scalers, train)
    pca = fit_pca(train_fused)
    assert np.isfinite(train_fused).all()
    assert np.isfinite(pca.transform(train_fused)).all()
    for label in ("validation", "test", "unassigned"):
        rows = {family: matrices[family][masks[label]] for family in matrices}
        fused = transform_and_fuse(scalers, rows)
        assert np.isfinite(fused).all()
        assert np.isfinite(pca.transform(fused)).all()


# ---------------------------------------------------------------------------
# PCA behaviour
# ---------------------------------------------------------------------------


def test_pca_learns_required_attributes():
    _counts, splits, _ids, matrices = _synthetic_dataset()
    masks = split_masks(splits)
    train = {family: matrices[family][masks["train"]] for family in matrices}
    scalers = fit_scalers(train)
    train_fused = transform_and_fuse(scalers, train)
    pca = fit_pca(train_fused)

    components = int(pca.n_components_)
    assert pca.components_.shape == (components, FUSED_DIMENSION)
    assert pca.explained_variance_.shape == (components,)
    assert pca.explained_variance_ratio_.shape == (components,)


def test_pca_reduces_dimensionality():
    _counts, splits, _ids, matrices = _synthetic_dataset()
    masks = split_masks(splits)
    train = {family: matrices[family][masks["train"]] for family in matrices}
    scalers = fit_scalers(train)
    pca = fit_pca(transform_and_fuse(scalers, train))
    assert int(pca.n_components_) < FUSED_DIMENSION


def test_pca_meets_explained_variance_target():
    _counts, splits, _ids, matrices = _synthetic_dataset()
    masks = split_masks(splits)
    train = {family: matrices[family][masks["train"]] for family in matrices}
    scalers = fit_scalers(train)
    pca = fit_pca(transform_and_fuse(scalers, train))
    assert float(pca.explained_variance_ratio_.sum()) >= TARGET_EXPLAINED_VARIANCE - 1e-6


# ---------------------------------------------------------------------------
# Data leakage: fitting must see training rows only
# ---------------------------------------------------------------------------


def test_scalers_fitted_on_training_rows_only():
    counts, splits, _ids, matrices = _synthetic_dataset()
    masks = split_masks(splits)
    train = {family: matrices[family][masks["train"]] for family in matrices}
    scalers = fit_scalers(train)

    for family in ("hog", "color_histogram", "lbp"):
        # each scaler saw exactly the training rows
        assert int(scalers[family].n_samples_seen_) == counts["train"]
        # canary: non-training rows are shifted by +25; a leaking scaler's
        # means would be dragged far away from the training means
        expected_train_mean = matrices[family][masks["train"]].mean(axis=0)
        assert np.allclose(scalers[family].mean_, expected_train_mean, atol=0.5)


def test_pca_fitted_on_training_rows_only():
    counts, splits, _ids, matrices = _synthetic_dataset()
    masks = split_masks(splits)
    train = {family: matrices[family][masks["train"]] for family in matrices}
    scalers = fit_scalers(train)
    train_fused = transform_and_fuse(scalers, train)
    pca = fit_pca(train_fused)

    # the principal axes are built from the training covariance: re-fitting
    # on training data alone must reproduce them exactly, whereas a fit that
    # had included the shifted non-training rows would not
    refit = PCA(
        n_components=TARGET_EXPLAINED_VARIANCE, svd_solver="full"
    ).fit(train_fused)
    assert np.allclose(np.abs(pca.components_), np.abs(refit.components_), atol=1e-6)


def test_validation_and_test_are_only_transformed():
    # transform path uses the fitted scalers/PCA and never mutates them
    _counts, splits, _ids, matrices = _synthetic_dataset()
    masks = split_masks(splits)
    train = {family: matrices[family][masks["train"]] for family in matrices}
    scalers = fit_scalers(train)
    train_fused = transform_and_fuse(scalers, train)
    pca = fit_pca(train_fused)
    components_before = pca.components_.copy()
    means_before = {family: scalers[family].mean_.copy() for family in scalers}

    rows = {family: matrices[family][masks["validation"]] for family in matrices}
    fused = transform_and_fuse(scalers, rows)
    pca.transform(fused)

    assert np.array_equal(pca.components_, components_before)
    for family, mean_before in means_before.items():
        assert np.array_equal(scalers[family].mean_, mean_before)


def test_transformation_is_deterministic():
    _counts, splits, _ids, matrices = _synthetic_dataset()
    masks = split_masks(splits)
    train = {family: matrices[family][masks["train"]] for family in matrices}
    scalers = fit_scalers(train)
    train_fused = transform_and_fuse(scalers, train)
    pca = fit_pca(train_fused)

    first = pca.transform(train_fused)
    second = pca.transform(transform_and_fuse(scalers, train))
    assert np.allclose(first, second, rtol=0, atol=1e-6)


# ---------------------------------------------------------------------------
# End-to-end run on synthetic data (run_fusion + artifact layout)
# ---------------------------------------------------------------------------


def test_run_fusion_end_to_end_on_synthetic(tmp_path):
    counts, splits, image_ids, matrices = _synthetic_dataset()
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    np.save(features_dir / "image_ids.npy", image_ids)
    np.save(features_dir / "splits.npy", splits)
    np.save(features_dir / "hog_features.npy", matrices["hog"])
    np.save(features_dir / "color_features.npy", matrices["color_histogram"])
    np.save(features_dir / "lbp_features.npy", matrices["lbp"])

    metadata = run_fusion(features_dir, tmp_path / "fusion")

    assert metadata["project_images"] == sum(counts.values())
    assert metadata["split_counts"] == counts
    assert metadata["input_dimensions"]["fused"] == 8206
    assert metadata["pca"]["components"] < FUSED_DIMENSION
    assert metadata["pca"]["explained_variance"] >= TARGET_EXPLAINED_VARIANCE - 1e-6
    assert metadata["standardization"]["train_samples_seen"] == {
        "hog": counts["train"],
        "color_histogram": counts["train"],
        "lbp": counts["train"],
    }

    fusion_dir = tmp_path / "fusion"
    for label in SPLIT_LABELS:
        features = np.load(fusion_dir / f"{label}_features.npy")
        ids = np.load(fusion_dir / f"{label}_image_ids.npy")
        assert len(features) == len(ids) == counts[label]  # alignment
        assert np.isfinite(features).all()
    expected_train_ids = image_ids[splits == "train"]
    assert np.array_equal(np.load(fusion_dir / "train_image_ids.npy"), expected_train_ids)
    assert (fusion_dir / "pca.joblib").exists()
    assert (fusion_dir / "scalers" / "hog_scaler.joblib").exists()

    reloaded_pca = joblib.load(fusion_dir / "pca.joblib")
    assert int(reloaded_pca.n_components_) == metadata["pca"]["components"]


# ---------------------------------------------------------------------------
# Integration: real Phase 3 inputs and real fusion artifacts
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not PHASE3_AVAILABLE, reason="data/features/*.npy not found — run Phase 3 first"
)
class TestPhase3Inputs:
    def test_input_dimensions_and_rows(self):
        features = load_phase3_features(FEATURES_DIR)
        assert len(features.image_ids) == 8091
        assert features.matrices["hog"].shape == (8091, 8100)
        assert features.matrices["color_histogram"].shape == (8091, 96)
        assert features.matrices["lbp"].shape == (8091, 10)
        assert dict(zip(*[a.tolist() for a in np.unique(features.splits, return_counts=True)])) == {
            "train": 6000,
            "validation": 1000,
            "test": 1000,
            "unassigned": 91,
        }


@pytest.mark.skipif(
    not FUSION_AVAILABLE,
    reason="data/features/fusion not found — run 'python -m src.features.fusion' first",
)
class TestRealFusionArtifacts:
    def test_shapes_and_alignment(self):
        metadata = json.loads((FUSION_DIR / "fusion_metadata.json").read_text(encoding="utf-8"))
        phase3 = load_phase3_features(FEATURES_DIR)
        reduced_dim = metadata["pca"]["components"]

        expected_counts = {"train": 6000, "validation": 1000, "test": 1000, "unassigned": 91}
        for label in SPLIT_LABELS:
            features = np.load(FUSION_DIR / f"{label}_features.npy")
            ids = np.load(FUSION_DIR / f"{label}_image_ids.npy")
            assert features.shape == (expected_counts[label], reduced_dim)
            assert len(ids) == expected_counts[label]
            assert features.dtype == np.float32
            assert np.isfinite(features).all()
            # rows correspond, in order, to the Phase 3 rows of that split
            expected_ids = phase3.image_ids[phase3.splits == label]
            assert np.array_equal(ids, expected_ids)

    def test_pca_model_and_metadata_agree(self):
        metadata = json.loads((FUSION_DIR / "fusion_metadata.json").read_text(encoding="utf-8"))
        pca = joblib.load(FUSION_DIR / "pca.joblib")
        assert int(pca.n_components_) == metadata["pca"]["components"]
        assert int(pca.n_components_) < FUSED_DIMENSION
        assert float(pca.explained_variance_ratio_.sum()) >= TARGET_EXPLAINED_VARIANCE - 1e-6
        assert np.isclose(
            float(pca.explained_variance_ratio_.sum()),
            metadata["pca"]["explained_variance"],
            atol=1e-6,
        )
        assert pca.components_.shape == (metadata["pca"]["components"], 8206)

    def test_scalers_were_fitted_on_training_rows_only(self):
        for family, filename in (
            ("hog", "hog_scaler.joblib"),
            ("color_histogram", "color_scaler.joblib"),
            ("lbp", "lbp_scaler.joblib"),
        ):
            scaler = joblib.load(FUSION_DIR / "scalers" / filename)
            assert int(scaler.n_samples_seen_) == 6000  # the train split only
            assert scaler.mean_.shape == (EXPECTED_INPUT_DIMENSIONS[family],)

    def test_metadata_counts(self):
        metadata = json.loads((FUSION_DIR / "fusion_metadata.json").read_text(encoding="utf-8"))
        assert metadata["project_images"] == 8091
        assert metadata["split_counts"] == {
            "train": 6000,
            "validation": 1000,
            "test": 1000,
            "unassigned": 91,
        }
        assert metadata["input_dimensions"] == {
            "hog": 8100,
            "color_histogram": 96,
            "lbp": 10,
            "fused": 8206,
        }
        assert metadata["standardization"]["fit_on"] == "train_only"
        assert metadata["pca"]["fit_on"] == "train_only"

    def test_phase3_outputs_untouched(self):
        features = load_phase3_features(FEATURES_DIR)
        assert features.matrices["hog"].shape == (8091, 8100)
        assert features.matrices["color_histogram"].shape == (8091, 96)
        assert features.matrices["lbp"].shape == (8091, 10)
        assert len(features.image_ids) == 8091
        assert len(features.splits) == 8091
