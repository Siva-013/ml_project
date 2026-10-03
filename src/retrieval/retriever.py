"""Phase 5 — classical image retrieval over the Phase 4 PCA space.

Given a query image, the retriever reuses the existing pipeline — Phase 2
preprocessing, Phase 3 handcrafted features, the Phase 4 training-fitted
scalers and PCA — to obtain a 1,804-dimensional query vector, then ranks
the 6,000-image **training-only** gallery by cosine similarity:

```
query image → Phase 2 preprocessing → HOG + Color + LBP
    → Phase 4 scalers (transform) → fusion → Phase 4 PCA (transform)
    → 1,804-D query vector
    → cosine similarity vs the precomputed, normalised gallery
    → deterministic Top-K ranking
```

Data-leakage rules:

* The gallery is **only** the training split (6,000 images); validation,
  test and unassigned images are never gallery members.
* Nothing is fitted here — the saved Phase 4 scalers and PCA are loaded and
  only ever used to transform. The query image never influences any fit.

Determinism: similarities are computed with a single vectorised dot
product; ranking sorts by similarity (descending) with image ID (ascending)
as the deterministic tie-breaker. Self-retrieval: by default, if the query
image is itself a gallery member, it is excluded from its own results
(``exclude_self=True``) — for validation/test queries this is a no-op
because those images are not in the gallery.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.features.extract import extract_features
from src.features.fusion import FUSION_SUBDIR, transform_and_fuse
from src.preprocessing.image_preprocessing import preprocess_image

_GALLERY_FEATURES_FILE = "train_features.npy"
_GALLERY_IDS_FILE = "train_image_ids.npy"
_SCALER_FILES = {
    "hog": "hog_scaler.joblib",
    "color_histogram": "color_scaler.joblib",
    "lbp": "lbp_scaler.joblib",
}


def l2_normalize(vectors: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return row-normalised copies (zero vectors are left as zeros)."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=axis, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


def cosine_similarities(
    gallery_normalized: np.ndarray, query_vector: np.ndarray
) -> np.ndarray:
    """Cosine similarity of one query against every normalised gallery row.

    ``cos(q, x) = (q · x) / (||q|| ||x||)`` — with both sides already
    L2-normalised this reduces to a single matrix-vector product,
    O(N × D) with N gallery rows and D dimensions.
    """
    query_normalized = l2_normalize(np.asarray(query_vector).reshape(1, -1))[0]
    return gallery_normalized @ query_normalized


def rank_results(
    similarities: np.ndarray,
    image_ids: np.ndarray,
    top_k: int,
    exclude_id: str | None = None,
) -> list[dict]:
    """Deterministically rank gallery rows by similarity.

    Primary key: similarity descending. Secondary key (exact ties): image
    ID ascending. ``exclude_id`` (the query image itself, when it is a
    gallery member) is removed before the Top-K cut.

    Returns:
        ``[{"image_id": str, "similarity": float}, ...]`` of length
        ``min(top_k, gallery_size)``.
    """
    order = np.lexsort((image_ids, -similarities))
    results = [
        {"image_id": str(image_ids[index]), "similarity": float(similarities[index])}
        for index in order
        if exclude_id is None or str(image_ids[index]) != exclude_id
    ]
    return results[:top_k]


def validate_top_k(top_k: int, gallery_size: int) -> int:
    """Validate ``1 <= top_k <= gallery_size`` and return it as an int.

    Raises:
        ValueError: For non-integer, zero, negative or too-large values.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)):
        raise ValueError(f"top_k must be an integer, got {type(top_k).__name__}")
    top_k = int(top_k)
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if top_k > gallery_size:
        raise ValueError(
            f"top_k must not exceed the gallery size ({gallery_size}), got {top_k}"
        )
    return top_k


class ImageRetriever:
    """Cosine-similarity retriever over the Phase 4 PCA training gallery.

    The gallery is loaded **once** at construction (Phase 4 outputs: the
    6,000 training rows of the PCA space plus their image IDs) and reused
    for any number of queries; gallery features are never recomputed.
    """

    def __init__(self, fusion_dir: str | Path | None = None) -> None:
        if fusion_dir is None:
            config = load_config()
            fusion_dir = resolve_path(config["paths"]["features_dir"]) / FUSION_SUBDIR
        self._fusion_dir = Path(fusion_dir)
        for filename in (_GALLERY_FEATURES_FILE, _GALLERY_IDS_FILE, "pca.joblib"):
            if not (self._fusion_dir / filename).exists():
                raise FileNotFoundError(
                    f"{self._fusion_dir / filename} not found — run "
                    "'python -m src.features.fusion' first"
                )

        self._gallery = np.load(self._fusion_dir / _GALLERY_FEATURES_FILE)
        self._gallery_ids = np.load(self._fusion_dir / _GALLERY_IDS_FILE)
        self._pca: PCA = joblib.load(self._fusion_dir / "pca.joblib")
        self._scalers: dict[str, StandardScaler] = {
            family: joblib.load(self._fusion_dir / "scalers" / filename)
            for family, filename in _SCALER_FILES.items()
        }

        if self._gallery.ndim != 2 or self._gallery.shape[0] != len(self._gallery_ids):
            raise ValueError(
                f"Gallery features {self._gallery.shape} and image IDs "
                f"({len(self._gallery_ids)},) are misaligned"
            )
        if self._gallery.shape[1] != self._pca.n_components_:
            raise ValueError(
                f"Gallery has {self._gallery.shape[1]} dimensions but the "
                f"loaded PCA expects {self._pca.n_components_}"
            )
        if not np.isfinite(self._gallery).all():
            raise ValueError("Gallery features contain non-finite values")
        self._gallery_normalized = l2_normalize(self._gallery)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def gallery_size(self) -> int:
        """Number of images in the retrieval gallery (the training split)."""
        return int(self._gallery.shape[0])

    @property
    def dimension(self) -> int:
        """Dimensionality of the retrieval space (PCA components)."""
        return int(self._gallery.shape[1])

    @property
    def gallery_image_ids(self) -> np.ndarray:
        """The gallery image IDs, aligned row-wise with the gallery matrix."""
        return self._gallery_ids

    # ------------------------------------------------------------------
    # Query pipeline (transform only — nothing is fitted here)
    # ------------------------------------------------------------------

    def extract_query_vector(self, query_image_path: str | Path) -> np.ndarray:
        """Map one image to its 1,804-D PCA query vector.

        Reuses Phase 2 preprocessing, Phase 3 feature extraction and the
        Phase 4 training-fitted scalers/PCA (transform only).
        """
        preprocessed = preprocess_image(query_image_path)
        features = extract_features(preprocessed)
        families = {
            "hog": features["hog"].reshape(1, -1),
            "color_histogram": features["color"].reshape(1, -1),
            "lbp": features["lbp"].reshape(1, -1),
        }
        fused = transform_and_fuse(self._scalers, families)
        return self._pca.transform(fused)

    def retrieve(
        self,
        query_image_path: str | Path,
        top_k: int = 5,
        exclude_self: bool = True,
    ) -> list[dict]:
        """Return the Top-K most visually similar gallery images.

        Args:
            query_image_path: Path to the query image.
            top_k: Number of results; ``1 <= top_k <= gallery_size``.
            exclude_self: When the query image is itself a gallery member
                (i.e. a training image), exclude it from its own results.
                A no-op for validation/test/unassigned queries.

        Returns:
            ``[{"image_id": str, "similarity": float}, ...]``, similarity
            descending, ties broken by image ID ascending.
        """
        top_k = validate_top_k(top_k, self.gallery_size)
        query_vector = self.extract_query_vector(query_image_path)
        similarities = cosine_similarities(self._gallery_normalized, query_vector[0])
        exclude_id = Path(query_image_path).name if exclude_self else None
        return rank_results(similarities, self._gallery_ids, top_k, exclude_id)
