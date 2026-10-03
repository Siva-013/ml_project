"""Phase 10 — caption-generation service.

Loads the existing Phase 4/5/6/7/8 artifacts **once** at startup (retriever
gallery, saved trigram, unigram/bigram fitted on the same training
captions, training-caption map) and serves caption requests through the
unchanged Phase 8 :class:`CaptionGenerator`. Nothing is trained or refit
here; the service is a thin, validated wrapper over the research pipeline.

Uploaded images live in a private temporary directory and are deleted
right after each request.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from src.captions.vocabulary import END, START
from src.caption_generation.candidates import load_training_captions
from src.caption_generation.generator import CaptionGenerator
from src.config import load_config, resolve_path
from src.evaluation.experiment import build_models
from src.features.fusion import FUSION_SUBDIR
from src.retrieval.retriever import ImageRetriever


class CaptionService:
    """Application-wide holder of the reusable pipeline artifacts."""

    def __init__(self) -> None:
        config = load_config()
        paths = config["paths"]
        captions_dir = resolve_path(paths["captions_dir"])
        self._images_dir = resolve_path(paths["images_dir"])
        self._fusion_dir = resolve_path(paths["features_dir"]) / FUSION_SUBDIR
        self.models = build_models(captions_dir)
        self.retriever = ImageRetriever(self._fusion_dir)
        self.training_captions = load_training_captions(captions_dir / "train_captions.csv")
        self._temp_dir = tempfile.mkdtemp(prefix="caption_upload_")
        self.gallery_ids: frozenset[str] = frozenset(
            str(image_id) for image_id in self.retriever.gallery_image_ids
        )

    # -- request handling -------------------------------------------------

    def generate(
        self,
        image_path: str | Path,
        *,
        top_k: int,
        model: str,
        max_length: int,
        min_length: int,
    ) -> dict:
        """Run the Phase 2–8 pipeline for one image and shape the response."""
        started = time.perf_counter()
        generator = CaptionGenerator(
            retriever=self.retriever,
            model=self.models[model],
            training_captions=self.training_captions,
            top_k=top_k,
            max_length=max_length,
            min_length=min_length,
        )
        result = generator.generate(image_path)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        candidate_sample = [
            " ".join(token for token in tokens if token not in (START, END))
            for tokens in generator.candidate_captions(result["retrieved"])[:5]
        ]
        return {
            "success": True,
            "generated_caption": result["caption"],
            "language_model": model,
            "top_k": top_k,
            "candidate_caption_count": result["candidate_caption_count"],
            "candidate_word_count": result["candidate_word_count"],
            "candidate_captions_sample": candidate_sample,
            "retrieved_images": [
                {
                    "image_id": retrieved["image_id"],
                    "similarity": float(retrieved["similarity"]),
                }
                for retrieved in result["retrieved"]
            ],
            "processing": {
                "feature_dimension": self.retriever.dimension,
                "similarity_metric": "cosine",
                "elapsed_ms": elapsed_ms,
            },
        }

    # -- gallery access (path-traversal safe) ------------------------------

    def gallery_image_path(self, image_id: str) -> Path:
        """Resolve a gallery image ID to a path, or raise ``LookupError``.

        Only IDs that belong to the known training gallery are accepted, so
        no client-supplied path component ever reaches the filesystem.
        """
        if image_id != Path(image_id).name or "/" in image_id or "\\" in image_id:
            raise LookupError("invalid image id")
        if image_id not in self.gallery_ids:
            raise LookupError(f"unknown gallery image: {image_id}")
        path = self._images_dir / image_id
        if not path.is_file():
            raise LookupError(f"gallery image not found on disk: {image_id}")
        return path

    # -- upload lifecycle ---------------------------------------------------

    def save_upload(self, data: bytes, suffix: str) -> Path:
        """Persist an upload in the private temp directory."""
        directory = Path(self._temp_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"upload_{time.time_ns()}{suffix}"
        path.write_bytes(data)
        return path

    def cleanup(self, *paths: Path | None) -> None:
        for path in paths:
            if path is not None:
                path.unlink(missing_ok=True)

    def shutdown(self) -> None:
        """Remove the temporary upload directory tree."""
        import shutil

        shutil.rmtree(self._temp_dir, ignore_errors=True)
