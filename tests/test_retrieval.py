"""Phase 5 tests: classical image retrieval.

Unit tests exercise the pure ranking/similarity helpers on small synthetic
arrays (no artifacts required). Integration tests load the real Phase 4
gallery and run real Flickr8k queries end to end (skipped until
``data/features/fusion`` exists).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config import load_config, resolve_path
from src.features.fusion import FUSION_SUBDIR
from src.retrieval.retriever import (
    ImageRetriever,
    cosine_similarities,
    l2_normalize,
    rank_results,
    validate_top_k,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FUSION_DIR = PROJECT_ROOT / "data" / "features" / FUSION_SUBDIR
FUSION_AVAILABLE = (FUSION_DIR / "pca.joblib").exists()


# ---------------------------------------------------------------------------
# Unit tests: ranking, similarity, validation (no artifacts required)
# ---------------------------------------------------------------------------


def test_rank_results_sorts_descending_and_breaks_ties_by_id():
    similarities = np.array([0.5, 0.5, 0.9, 0.1])
    image_ids = np.array(["b.jpg", "a.jpg", "c.jpg", "d.jpg"])
    results = rank_results(similarities, image_ids, top_k=4)
    assert [r["image_id"] for r in results] == ["c.jpg", "a.jpg", "b.jpg", "d.jpg"]
    assert [r["similarity"] for r in results] == [0.9, 0.5, 0.5, 0.1]


def test_rank_results_excludes_query_id_before_top_k_cut():
    similarities = np.array([0.99, 0.8, 0.7])
    image_ids = np.array(["query.jpg", "x.jpg", "y.jpg"])
    results = rank_results(similarities, image_ids, top_k=2, exclude_id="query.jpg")
    assert [r["image_id"] for r in results] == ["x.jpg", "y.jpg"]  # query removed first


def test_cosine_similarities_known_values():
    gallery = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    gallery_normalized = l2_normalize(gallery)
    assert np.allclose(cosine_similarities(gallery_normalized, [2.0, 0.0]), [1.0, 0.0])
    assert np.allclose(
        cosine_similarities(gallery_normalized, [1.0, 1.0]), [2 ** -0.5, 2 ** -0.5], atol=1e-6
    )


def test_cosine_similarity_symmetry_and_bounds_on_synthetic():
    rng = np.random.default_rng(3)
    gallery = rng.normal(size=(50, 16)).astype(np.float32)
    query = rng.normal(size=16).astype(np.float32)
    sims = cosine_similarities(l2_normalize(gallery), query)
    assert sims.shape == (50,)
    assert np.all(sims <= 1.0 + 1e-6)
    assert np.all(sims >= -1.0 - 1e-6)


def test_l2_normalize_handles_zero_rows():
    normalized = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))
    assert np.allclose(normalized[0], [0.6, 0.8])
    assert np.allclose(normalized[1], [0.0, 0.0])  # zero rows stay zero


def test_validate_top_k_rejects_invalid_values():
    assert validate_top_k(5, 100) == 5
    with pytest.raises(ValueError, match=">= 1"):
        validate_top_k(0, 100)
    with pytest.raises(ValueError, match=">= 1"):
        validate_top_k(-3, 100)
    with pytest.raises(ValueError, match="exceed"):
        validate_top_k(101, 100)
    with pytest.raises(ValueError, match="integer"):
        validate_top_k("5", 100)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Integration: real Phase 4 gallery, real Flickr8k queries
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def retriever() -> ImageRetriever:
    return ImageRetriever(FUSION_DIR)


@pytest.fixture(scope="module")
def images_dir() -> Path:
    config = load_config()
    return resolve_path(config["paths"]["images_dir"])


@pytest.mark.skipif(
    not FUSION_AVAILABLE, reason="fusion artifacts missing"
)
class TestRealGallery:
    """Structural correctness of the gallery (spec tests 1, 2, 4, 12)."""

    def test_gallery_dimensions(self, retriever: ImageRetriever):
        assert retriever.gallery_size == 6000
        assert retriever.dimension == 1804

    def test_gallery_id_alignment(self, retriever: ImageRetriever):
        features = np.load(FUSION_DIR / "train_features.npy", mmap_mode="r")
        assert len(features) == len(retriever.gallery_image_ids)

    def test_gallery_values_finite(self, retriever: ImageRetriever):
        assert np.isfinite(retriever._gallery).all()

    def test_gallery_contains_training_split_only(self, retriever: ImageRetriever):
        from src.features.fusion import load_phase3_features

        phase3 = load_phase3_features(PROJECT_ROOT / "data" / "features")
        expected = phase3.image_ids[phase3.splits == "train"]
        assert np.array_equal(retriever.gallery_image_ids, expected)
        non_training = phase3.image_ids[phase3.splits != "train"]
        assert not (set(non_training.tolist()) & set(retriever.gallery_image_ids.tolist()))


@pytest.mark.skipif(
    not FUSION_AVAILABLE, reason="fusion artifacts missing"
)
class TestRealQueries:
    """End-to-end retrieval behaviour on real Flickr8k images."""

    def test_query_representation_is_1804d_and_finite(
        self, retriever: ImageRetriever, images_dir: Path
    ):
        query_id = str(retriever.gallery_image_ids[0])
        vector = retriever.extract_query_vector(images_dir / query_id)
        assert vector.shape == (1, 1804)
        assert np.isfinite(vector).all()

    def test_top_k_returns_exactly_k_results(
        self, retriever: ImageRetriever, images_dir: Path
    ):
        query_id = str(retriever.gallery_image_ids[1])
        results = retriever.retrieve(images_dir / query_id, top_k=5)
        assert len(results) == 5
        assert all(set(result) == {"image_id", "similarity"} for result in results)

    def test_similarity_within_cosine_bounds(
        self, retriever: ImageRetriever, images_dir: Path
    ):
        query_id = str(retriever.gallery_image_ids[1])
        results = retriever.retrieve(images_dir / query_id, top_k=10)
        for result in results:
            assert -1.0 - 1e-6 <= result["similarity"] <= 1.0 + 1e-6

    def test_similarities_are_descending(
        self, retriever: ImageRetriever, images_dir: Path
    ):
        query_id = str(retriever.gallery_image_ids[2])
        similarities = [
            result["similarity"]
            for result in retriever.retrieve(images_dir / query_id, top_k=25)
        ]
        assert all(
            similarities[index] >= similarities[index + 1]
            for index in range(len(similarities) - 1)
        )

    def test_retrieval_is_deterministic(
        self, retriever: ImageRetriever, images_dir: Path
    ):
        query_id = str(retriever.gallery_image_ids[3])
        first = retriever.retrieve(images_dir / query_id, top_k=10)
        second = retriever.retrieve(images_dir / query_id, top_k=10)
        assert [r["image_id"] for r in first] == [r["image_id"] for r in second]
        assert np.allclose(
            [r["similarity"] for r in first],
            [r["similarity"] for r in second],
            rtol=0,
            atol=0.0,
        )

    def test_invalid_top_k_raises_clear_errors(
        self, retriever: ImageRetriever, images_dir: Path
    ):
        query_id = str(retriever.gallery_image_ids[0])
        with pytest.raises(ValueError, match=">= 1"):
            retriever.retrieve(images_dir / query_id, top_k=0)
        with pytest.raises(ValueError, match="exceed"):
            retriever.retrieve(images_dir / query_id, top_k=retriever.gallery_size + 1)

    def test_self_exclusion_for_training_query(
        self, retriever: ImageRetriever, images_dir: Path
    ):
        query_id = str(retriever.gallery_image_ids[0])
        results = retriever.retrieve(images_dir / query_id, top_k=10, exclude_self=True)
        assert query_id not in [r["image_id"] for r in results]
        # with exclusion disabled, the query retrieves itself with ~1.0 similarity
        inclusive = retriever.retrieve(
            images_dir / query_id, top_k=1, exclude_self=False
        )
        assert inclusive[0]["image_id"] == query_id
        assert inclusive[0]["similarity"] > 0.999

    def test_real_validation_query(self, retriever: ImageRetriever, images_dir: Path):
        from src.features.fusion import load_phase3_features

        phase3 = load_phase3_features(PROJECT_ROOT / "data" / "features")
        query_id = str(phase3.image_ids[phase3.splits == "validation"][0])
        results = retriever.retrieve(images_dir / query_id, top_k=5)
        assert len(results) == 5
        # validation images are not gallery members: nothing to exclude
        assert all(0.0 <= r["similarity"] <= 1.0 + 1e-6 for r in results)
