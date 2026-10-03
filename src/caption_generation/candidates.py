"""Phase 8 — candidate captions and candidate words from retrieval results.

The visual side of the project never "understands" the image: it only
identifies visually similar **training** images (Phase 5, training-only
gallery). This module turns those retrieval results into the language
side's raw material — the training captions of the retrieved images — and
derives from them a deterministic candidate-word list for the decoder.

Data-leakage contract: candidate captions come exclusively from the
training split (the Phase 5 gallery is training-only, and the caption map
loaded here is built from ``train_captions.csv`` only). Validation and test
captions are never candidates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from src.captions.vocabulary import PAD, START

#: Tokens that may never be generated as ordinary words.
FORBIDDEN_WORDS: frozenset[str] = frozenset({START, PAD})


def load_training_captions(csv_path: str | Path) -> dict[str, list[list[str]]]:
    """Map each training image ID to its Phase 6 token sequences.

    The map is built from ``train_captions.csv`` only (5 captions per
    image, each including the ``<START>``/``<END>`` boundary tokens), and
    insertion order follows the CSV — deterministic.
    """
    frame = pd.read_csv(csv_path)
    if not {"image_id", "tokens"}.issubset(frame.columns):
        raise ValueError(f"{csv_path} lacks the required columns image_id/tokens")
    captions: dict[str, list[list[str]]] = {}
    for image_id, tokens_json in zip(frame["image_id"], frame["tokens"]):
        captions.setdefault(str(image_id), []).append(json_load_tokens(tokens_json))
    return captions


def json_load_tokens(tokens_json: str) -> list[str]:
    """Parse the documented ``tokens`` CSV format (a JSON array string)."""
    import json

    return json.loads(tokens_json)


def collect_candidate_captions(
    retrieval_results: Sequence[Mapping],
    training_captions: Mapping[str, Sequence[Sequence[str]]],
) -> list[list[str]]:
    """Gather the training captions of the retrieved images.

    Args:
        retrieval_results: Phase 5 results
            (``[{"image_id": ..., "similarity": ...}, ...]``, ranked).
        training_captions: Map from :func:`load_training_captions`.

    Returns:
        Candidate token sequences in deterministic order: by retrieval
        rank first, then by the image's caption order in the CSV
        (caption_id ascending).

    Raises:
        KeyError: If a retrieved image has no training captions — the
            gallery is training-only, so this would indicate a
            misconfigured pipeline rather than something to skip silently.
    """
    candidates: list[list[str]] = []
    for result in retrieval_results:
        image_id = str(result["image_id"])
        if image_id not in training_captions:
            raise KeyError(
                f"Retrieved image '{image_id}' has no training captions — "
                "the gallery must contain training images only"
            )
        candidates.extend([list(tokens) for tokens in training_captions[image_id]])
    return candidates


def build_candidate_words(
    candidate_captions: Sequence[Sequence[str]],
    forbidden: frozenset[str] = FORBIDDEN_WORDS,
) -> list[str]:
    """Deterministic candidate-word list for the decoder.

    Words are the unique tokens of the candidate captions (``<START>`` and
    ``<PAD>`` excluded; ``<END>`` is kept — choosing it is how the decoder
    stops). Ordering: frequency within the candidate captions descending,
    ties alphabetical. The decoder must not depend on any other order.
    """
    counts: dict[str, int] = {}
    for caption in candidate_captions:
        for token in caption:
            if token in forbidden:
                continue
            counts[token] = counts.get(token, 0) + 1
    return [token for token, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
