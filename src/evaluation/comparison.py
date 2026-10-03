"""Phase 9 — comparison tables for language models and Top-K values."""

from __future__ import annotations

import pandas as pd

METRIC_COLUMNS = (
    "bleu_1",
    "bleu_2",
    "bleu_3",
    "bleu_4",
    "rouge_1",
    "rouge_2",
    "rouge_l",
    "average_caption_length",
    "unk_count",
)


def model_comparison_frame(model_metrics: dict[str, dict], perplexities: dict[str, float] | None = None) -> pd.DataFrame:
    """One row per language model: perplexity + the full metric panel.

    Perplexity and caption metrics measure different properties (how well
    the model predicts held-out text vs. how well the *generated* captions
    match human references); they are reported side by side, never merged
    into a single score.
    """
    rows = []
    for name in sorted(model_metrics):
        row = {"model": name}
        if perplexities and name in perplexities:
            row["validation_perplexity"] = perplexities[name]
        row.update({column: model_metrics[name][column] for column in METRIC_COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows)


def top_k_comparison_frame(top_k_metrics: dict[int, dict]) -> pd.DataFrame:
    """One row per Top-K value: the full metric panel + failure rate."""
    rows = []
    for k in sorted(top_k_metrics):
        metrics = top_k_metrics[k]
        rows.append({"top_k": k, **{column: metrics[column] for column in METRIC_COLUMNS}})
    return pd.DataFrame(rows)
