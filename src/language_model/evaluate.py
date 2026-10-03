"""Phase 7 — evaluation utilities for the n-gram language model.

Perplexity is the primary quality measure:

    PP = exp( -(1/N) * Σ log P(w_i | context_i) )

Lower is better: a model that predicts the held-out captions more
confidently has lower perplexity. This module evaluates **fitted** models
on caption sets (validation in Phase 7; test stays reserved for Phase 9).
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from src.language_model.ngram import NGramLanguageModel
from src.language_model.smoothing import LaplaceSmoothing


def prediction_count(captions: Iterable[Sequence[str]]) -> int:
    """Number of predicted tokens (all tokens except each caption's ``<START>``)."""
    return sum(max(len(sequence) - 1, 0) for sequence in captions)


def evaluate_perplexity(
    model: NGramLanguageModel, captions: Iterable[Sequence[str]]
) -> dict:
    """Evaluate ``model`` on a caption set (e.g. the validation split).

    Returns:
        ``{"perplexity", "prediction_count", "mean_log_probability"}``.
        ``mean_log_probability`` is the natural-log probability per
        predicted token (always ≤ 0; closer to 0 is better).
    """
    sequences = [sequence for sequence in captions]
    total_log_probability = sum(
        model.sentence_log_probability(sequence)
        for sequence in sequences
        if len(sequence) > 1
    )
    count = prediction_count(sequences)
    if count == 0:
        raise ValueError("Cannot evaluate a caption set with zero predictions")
    mean_log_probability = total_log_probability / count
    return {
        "perplexity": math.exp(-mean_log_probability),
        "prediction_count": count,
        "mean_log_probability": mean_log_probability,
    }


def compare_orders(
    train_captions: Iterable[Sequence[str]],
    evaluation_captions: Iterable[Sequence[str]],
    *,
    orders: tuple[int, ...] = (1, 2, 3),
    alpha: float = 1.0,
    vocabulary_tokens: set[str] | None = None,
) -> dict[int, dict]:
    """Fit one model per order on the same training captions and compare.

    All models are fitted on ``train_captions`` only; ``evaluation_captions``
    (the validation split in Phase 7) are only ever scored.

    Returns:
        Mapping ``order → {"perplexity", "prediction_count",
        "mean_log_probability", "ngram_counts"}`` where ``ngram_counts``
        lists the number of distinct fitted n-grams per order.
    """
    train_sequences = [sequence for sequence in train_captions]
    results: dict[int, dict] = {}
    for order in orders:
        model = NGramLanguageModel(
            n=order,
            smoothing=LaplaceSmoothing(alpha=alpha),
            vocabulary_tokens=vocabulary_tokens,
        ).fit(train_sequences)
        metrics = evaluate_perplexity(model, evaluation_captions)
        metrics["ngram_counts"] = {
            inner_order: len(model.ngram_counts[inner_order])
            for inner_order in range(1, order + 1)
        }
        results[order] = metrics
    return results
