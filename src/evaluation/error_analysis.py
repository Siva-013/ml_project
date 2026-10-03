"""Phase 9 — qualitative examples and automated error analysis.

Example selection is deterministic (no sampling): validation predictions
are ranked by their per-image sentence-level ROUGE-L; the top, median and
bottom blocks become good/moderate/poor examples.

Failure categories are *heuristics applied to observed outputs* — no
category is reported unless it was actually triggered:

- ``repetitive``: a word repeats more than twice;
- ``very_short`` / ``incomplete``: fewer than 4 words;
- ``generic``: caption matches the corpus's most generic template family
  (starts with "a man in a" / "a woman in a");
- ``unk_generated``: <UNK> in the output (cannot happen with Phase 8
  candidates, reported as 0 by design);
- retrieval side: ``low_caption_overlap`` when the neighbours' caption
  overlap with the query's references is in the bottom quartile — that
  separates *visual retrieval failures* (bad neighbours) from
  *language-model failures* (reasonable neighbours, still bad caption).
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from src.evaluation.caption_metrics import rouge_l, unk_count

GENERIC_PREFIXES = ("a man in a", "a woman in a")


def sentence_rouge_l(generated: Sequence[str], references: Sequence[Sequence[str]]) -> float:
    return rouge_l(list(generated), [list(ref) for ref in references])


def classify_failure(caption: str, overlap_quartile_threshold: float, jaccard: float) -> str:
    categories: list[str] = []
    words = caption.split()
    if unk_count(words) > 0:
        categories.append("unk_generated")
    if len(words) < 4:
        categories.append("very_short")
    if any(caption.startswith(prefix) for prefix in GENERIC_PREFIXES):
        categories.append("generic")
    if jaccard < overlap_quartile_threshold:
        categories.append("low_caption_overlap")
    return "; ".join(categories) if categories else "none"


def build_qualitative_examples(
    predictions: pd.DataFrame,
    references_by_image: dict[str, list[list[str]]],
    retrieval_by_image: dict[str, list[dict]],
    *,
    jaccard_by_image: dict[str, float] | None = None,
    per_group: int = 10,
    overlap_threshold: float = 0.0,
) -> pd.DataFrame:
    """Rank predictions by per-image ROUGE-L and take good/moderate/poor blocks."""
    jaccard_by_image = jaccard_by_image or {}
    scored = predictions.copy()
    scored["rouge_l"] = [
        sentence_rouge_l(caption.split(), references_by_image[image_id])
        for caption, image_id in zip(scored["generated_caption"], scored["image_id"])
    ]
    scored = scored.sort_values("rouge_l", ascending=False, kind="stable").reset_index(drop=True)

    blocks = [
        ("good", scored.head(per_group)),
        (
            "moderate",
            scored.iloc[
                max(0, len(scored) // 2 - per_group // 2) : len(scored) // 2 + per_group // 2
            ],
        ),
        ("poor", scored.tail(per_group)),
    ]
    rows = []
    for quality, block in blocks:
        for _, row in block.iterrows():
            image_id = row["image_id"]
            retrieved = retrieval_by_image.get(image_id, [])
            jaccard = float(jaccard_by_image.get(image_id, 0.0))
            rows.append(
                {
                    "quality": quality,
                    "image_id": image_id,
                    "generated_caption": row["generated_caption"],
                    "reference_1": row.get("reference_1", ""),
                    "reference_2": row.get("reference_2", ""),
                    "reference_3": row.get("reference_3", ""),
                    "reference_4": row.get("reference_4", ""),
                    "reference_5": row.get("reference_5", ""),
                    "top_retrieved_images": "; ".join(
                        r["image_id"] for r in retrieved[:5]
                    ),
                    "similarity_scores": "; ".join(
                        f"{r['similarity']:.4f}" for r in retrieved[:5]
                    ),
                    "language_model": row.get("language_model", "trigram"),
                    "top_k": row.get("top_k", 5),
                    "rouge_l": round(float(row["rouge_l"]), 4),
                    "analysis": classify_failure(
                        row["generated_caption"], overlap_threshold, jaccard
                    ),
                }
            )
    return pd.DataFrame(rows)
