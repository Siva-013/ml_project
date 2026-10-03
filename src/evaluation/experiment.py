"""Phase 9 — experiment orchestration for the caption-generation pipeline.

One pass over a split answers several experiments at once with a single
retrieval per image:

* per-image predictions for the primary configuration (trigram, K = 5);
* language-model comparison (unigram / bigram / trigram decoded on the
  same retrieved evidence);
* Top-K comparison (K ∈ {1, 3, 5, 10}) by slicing the K = 10 ranking —
  identical to retrieving each K directly because cosine ranking is
  deterministic;
* retrieval-proxy statistics (caption-overlap of the neighbours).

The models are the Phase 7 models: the saved trigram, plus unigram/bigram
fitted here on the same 30,000 training captions. Validation/test captions
are only ever *scored*, never fitted. Decoding reuses Phase 8's
``greedy_decode`` so the evaluated system is exactly the deployed system.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd

from src.caption_generation.candidates import (
    collect_candidate_captions,
    load_training_captions,
)
from src.caption_generation.generator import greedy_decode
from src.config import load_config, resolve_path
from src.evaluation.caption_metrics import evaluate_captions, strip_boundary_tokens, unk_count
from src.evaluation.retrieval_metrics import retrieval_relevance
from src.features.fusion import FUSION_SUBDIR
from src.language_model.ngram import NGramLanguageModel, load_model
from src.language_model.smoothing import LaplaceSmoothing
from src.retrieval.retriever import ImageRetriever

EVALUATION_TOP_K_VALUES: tuple[int, ...] = (1, 3, 5, 10)
MAX_TOP_K = max(EVALUATION_TOP_K_VALUES)
MODEL_ORDER = {"unigram": 1, "bigram": 2, "trigram": 3}


def load_split_captions(captions_dir: Path, split: str) -> pd.DataFrame:
    """Load one split's Phase 6 CSV (image_id, caption_id, clean_caption, …)."""
    path = Path(captions_dir) / f"{split}_captions.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run Phase 6 first")
    return pd.read_csv(path)


def load_training_token_sequences(captions_dir: Path) -> list[list[str]]:
    """Training token sequences (with boundaries) for fitting n-gram models."""
    frame = pd.read_csv(Path(captions_dir) / "train_captions.csv")
    return [json.loads(tokens) for tokens in frame["tokens"]]


def build_models(
    captions_dir: Path, alpha: float = 1.0
) -> dict[str, NGramLanguageModel]:
    """Fit unigram/bigram here; trigram is the saved Phase 7 model."""
    vocabulary_path = captions_dir / "vocabulary.json"
    vocabulary_data = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    vocabulary_tokens = set(vocabulary_data["token_to_id"])
    train_sequences = load_training_token_sequences(captions_dir)
    models: dict[str, NGramLanguageModel] = {
        "trigram": load_model(
            resolve_path(config_paths()["language_model_dir"]) / "ngram_model.json",
            vocabulary_tokens=vocabulary_tokens,
        )
    }
    for name, order in (("unigram", 1), ("bigram", 2)):
        models[name] = (
            NGramLanguageModel(
                n=order,
                smoothing=LaplaceSmoothing(alpha=alpha),
                vocabulary_tokens=vocabulary_tokens,
            ).fit(train_sequences)
        )
    return models


def config_paths() -> dict:
    config = load_config()
    return config["paths"]


def reference_groups(split_frame: pd.DataFrame) -> tuple[list[str], list[list[list[str]]]]:
    """Per-image ordered IDs and their five reference token lists (no boundaries)."""
    frame = split_frame.sort_values(["image_id", "caption_id"])
    image_ids: list[str] = []
    references: list[list[list[str]]] = []
    for image_id, group in frame.groupby("image_id", sort=False):
        image_ids.append(str(image_id))
        references.append(
            [strip_boundary_tokens(json.loads(tokens)) for tokens in group["tokens"]]
        )
    return image_ids, references


def run_split_experiment(
    split: str,
    *,
    models: dict[str, NGramLanguageModel] | None = None,
    top_k_values: tuple[int, ...] = EVALUATION_TOP_K_VALUES,
    primary_model: str = "trigram",
    primary_top_k: int = 5,
    max_images: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Evaluate the pipeline on one split (validation or test).

    One retrieval per image (at the largest requested K) feeds every
    (model, K) decoding combination. Nothing is fitted on the split.

    Returns:
        ``{
          "predictions": DataFrame for the primary configuration,
          "retrieval_rows": list[dict] per-query/per-rank retrieval rows,
          "model_metrics": {model: metrics dict},
          "top_k_metrics": {k: metrics dict},
          "retrieval_summary": {k: {"mean_similarity", "mean_jaccard",
                                    "mean_containment", "exact_self_matches"}},
          "primary_metrics": metrics dict of (primary_model, primary_top_k),
        }``

    Raises:
        ValueError: If ``split`` is not validation/test, or the primary
            configuration is not among the evaluated combinations.
    """
    if split not in ("validation", "test"):
        raise ValueError(f"split must be 'validation' or 'test', got {split!r}")
    paths = config_paths()
    captions_dir = resolve_path(paths["captions_dir"])
    if models is None:
        models = build_models(captions_dir)
    if primary_model not in models:
        raise ValueError(f"primary model {primary_model!r} not among {sorted(models)}")

    retriever = ImageRetriever(resolve_path(paths["features_dir"]) / FUSION_SUBDIR)
    training_captions = load_training_captions(captions_dir / "train_captions.csv")
    split_frame = load_split_captions(captions_dir, split)
    image_ids, references = reference_groups(split_frame)
    if max_images is not None:
        image_ids = image_ids[:max_images]
        references = references[:max_images]
    largest_k = max(max(top_k_values), primary_top_k)

    prediction_rows: list[dict] = []
    retrieval_rows: list[dict] = []
    model_runs: dict[str, tuple[list[list[str]], list[list[list[str]]]]] = {
        name: ([], []) for name in models
    }
    top_k_runs: dict[int, tuple[list[list[str]], list[list[list[str]]]]] = {
        k: ([], []) for k in top_k_values
    }
    retrieval_stats: dict[int, dict[str, list[float]]] = {
        k: {"mean_similarity": [], "mean_jaccard": [], "mean_containment": []}
        for k in top_k_values
    }
    jaccard_by_image: dict[str, float] = {}
    exact_self_matches = 0

    for index, image_id in enumerate(image_ids):
        results = retriever.retrieve(
            resolve_path(paths["images_dir"]) / image_id, top_k=largest_k
        )
        if any(str(r["image_id"]) == image_id for r in results):
            exact_self_matches += 1  # cannot happen for validation/test by design

        for k in top_k_values:
            top_results = results[:k]
            retrieved_captions = collect_candidate_captions(top_results, training_captions)
            proxies = retrieval_relevance(references[index], retrieved_captions)
            if k == primary_top_k:
                jaccard_by_image[image_id] = proxies["jaccard_overlap"]
            similarity = sum(r["similarity"] for r in top_results) / k
            retrieval_stats[k]["mean_similarity"].append(similarity)
            retrieval_stats[k]["mean_jaccard"].append(proxies["jaccard_overlap"])
            retrieval_stats[k]["mean_containment"].append(proxies["containment"])
            if split == "validation" or k <= 5:
                for rank, r in enumerate(results[:k], start=1):
                    retrieval_rows.append(
                        {
                            "image_id": image_id,
                            "split": split,
                            "rank": rank,
                            "retrieved_image_id": r["image_id"],
                            "similarity": r["similarity"],
                        }
                    )

        for name, model in models.items():
            for k in top_k_values:
                if name != primary_model and k != primary_top_k:
                    continue  # full grid only for the primary model
                candidate_captions = collect_candidate_captions(
                    results[:k], training_captions
                )
                decoded = greedy_decode(model, candidate_captions)
                tokens = decoded["tokens"]
                model_runs[name][0].append(tokens)
                model_runs[name][1].append(references[index])
                if name == primary_model:
                    top_k_runs[k][0].append(tokens)
                    top_k_runs[k][1].append(references[index])

        primary_tokens = model_runs[primary_model][0][-1]
        prediction_rows.append(
            {
                "image_id": image_id,
                "generated_caption": " ".join(primary_tokens),
                **{
                    f"reference_{i + 1}": " ".join(references[index][i])
                    for i in range(len(references[index]))
                },
                "top_k": primary_top_k,
                "language_model": primary_model,
                "caption_length": len(primary_tokens),
                "unk_count": unk_count(primary_tokens),
            }
        )
        if progress is not None:
            progress(index + 1, len(image_ids))

    model_metrics = {
        name: evaluate_captions(refs, hyps)
        for name, (hyps, refs) in model_runs.items()
    }
    top_k_metrics = {
        k: evaluate_captions(refs, hyps) for k, (hyps, refs) in top_k_runs.items()
    }
    primary_metrics = top_k_metrics[primary_top_k]
    retrieval_summary = {
        k: {
            key: sum(values) / len(values) if values else 0.0
            for key, values in stats.items()
        }
        | {"exact_self_matches": exact_self_matches}
        for k, stats in retrieval_stats.items()
    }
    return {
        "predictions": pd.DataFrame(prediction_rows),
        "retrieval_rows": retrieval_rows,
        "jaccard_by_image": jaccard_by_image,
        "model_metrics": model_metrics,
        "top_k_metrics": top_k_metrics,
        "retrieval_summary": retrieval_summary,
        "primary_metrics": primary_metrics,
    }
