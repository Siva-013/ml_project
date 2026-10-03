"""Phase 8 — caption generation by greedy n-gram decoding over retrieval evidence.

This is where the project's two branches join — without any learning:

```
query image → Phase 2 → Phase 3 → Phase 4 → Phase 5 retrieval
    → Top-K visually similar TRAINING images
    → their training captions (candidate captions, ≤ 5·K)
    → candidate words (+ <END>)
    → trigram LM score + retrieval-evidence bonus (greedy, deterministic)
    → generated caption
```

**Scoring.** At every step each candidate word ``w`` is scored as

    score(w | context) = log P(w | context)            (Phase 7 trigram, Laplace)
                       + visual_weight · log evidence(w)

where ``evidence(w)`` is the fraction of candidate captions containing
``w`` (so words echoed by many retrieved neighbours are boosted, words
appearing in a single retrieved caption are penalised, and ``<END>`` —
present in every caption — gets weight 0). ``visual_weight`` scales the
retrieval evidence; 0 reduces the decoder to the pure language model.

**Constraints.** Decoding is greedy (highest score wins; ties resolved by
the deterministic candidate order). ``<START>``/``<PAD>`` are never
generated; an immediate word repetition is banned; a word may appear at
most ``max_word_repetitions`` times; ``<END>`` is only eligible once
``min_length`` words exist. Generation stops at ``<END>`` or
``max_length``; the displayed caption never contains boundary tokens.

Everything is deterministic: same image, same artifacts, same parameters →
identical caption.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

from src.captions.vocabulary import END, PAD, START
from src.caption_generation.candidates import (
    build_candidate_words,
    collect_candidate_captions,
    load_training_captions,
)
from src.config import load_config, resolve_path
from src.features.fusion import FUSION_SUBDIR
from src.language_model.ngram import NGramLanguageModel, load_model
from src.language_model.train import load_phase6_captions, load_phase6_vocabulary
from src.retrieval.retriever import ImageRetriever

DEFAULT_TOP_K = 5
DEFAULT_MAX_LENGTH = 20
DEFAULT_MIN_LENGTH = 3
DEFAULT_VISUAL_WEIGHT = 1.0
DEFAULT_MAX_WORD_REPETITIONS = 2


def greedy_decode(
    model: NGramLanguageModel,
    candidate_captions: Sequence[Sequence[str]],
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
    min_length: int = DEFAULT_MIN_LENGTH,
    visual_weight: float = DEFAULT_VISUAL_WEIGHT,
    max_word_repetitions: int = DEFAULT_MAX_WORD_REPETITIONS,
) -> dict:
    """Deterministic greedy decoding over the candidate words.

    Shared by :class:`CaptionGenerator` and the Phase 9 evaluation so both
    use exactly the same decoding rules. See the module docstring for the
    scoring formula and the constraint set.

    Returns:
        ``{"tokens", "caption", "candidate_word_count",
        "candidate_caption_count"}`` — boundary tokens removed.
    """
    candidate_words = build_candidate_words(candidate_captions)
    total_captions = len(candidate_captions)
    occurrences: dict[str, int] = {}
    for caption in candidate_captions:
        for token in set(caption):
            occurrences[token] = occurrences.get(token, 0) + 1

    tokens: list[str] = [START]
    chosen: list[str] = []
    while len(chosen) < max_length:
        allowed = [
            word
            for word in candidate_words
            if word != tokens[-1]  # no immediate repetition
            and chosen.count(word) < max_word_repetitions
        ]
        if len(chosen) < min_length:
            allowed = [word for word in allowed if word != END]
        if not allowed:
            break  # candidate pool exhausted; deterministic early stop

        best_word: str | None = None
        best_score = -math.inf
        for word in allowed:
            evidence = occurrences.get(word, 0) / total_captions
            context = tuple(tokens[-(model.n - 1) :]) if model.n > 1 else ()
            score = model.get_log_probability(context, word)
            score += visual_weight * math.log(evidence)
            if score > best_score:  # strict > keeps the earlier candidate on ties
                best_score, best_word = score, word
        if best_word is None or best_word == END:
            break
        tokens.append(best_word)
        chosen.append(best_word)

    return {
        "tokens": tokens[1:],  # boundary tokens removed from the output
        "caption": " ".join(tokens[1:]),
        "candidate_word_count": len(candidate_words),
        "candidate_caption_count": total_captions,
    }


class CaptionGenerator:
    """Deterministic greedy caption generator over retrieval evidence.

    Args:
        retriever: Phase 5 :class:`ImageRetriever` (loaded from the Phase 4
            fusion directory by default).
        model: Phase 7 :class:`NGramLanguageModel` (loaded from
            ``data/language_model/ngram_model.json`` by default; never
            refit here).
        training_captions: Map image ID → token sequences (Phase 6
            ``train_captions.csv`` only, by default).
        top_k / max_length / min_length / visual_weight /
            max_word_repetitions: decoding parameters.
    """

    def __init__(
        self,
        retriever: ImageRetriever | None = None,
        model: NGramLanguageModel | None = None,
        training_captions: Mapping[str, Sequence[Sequence[str]]] | None = None,
        *,
        top_k: int = DEFAULT_TOP_K,
        max_length: int = DEFAULT_MAX_LENGTH,
        min_length: int = DEFAULT_MIN_LENGTH,
        visual_weight: float = DEFAULT_VISUAL_WEIGHT,
        max_word_repetitions: int = DEFAULT_MAX_WORD_REPETITIONS,
    ) -> None:
        if max_length < 1:
            raise ValueError(f"max_length must be >= 1, got {max_length}")
        if not 0 <= min_length <= max_length:
            raise ValueError(
                f"min_length must satisfy 0 <= min_length <= max_length, "
                f"got {min_length} and {max_length}"
            )
        if visual_weight < 0:
            raise ValueError(f"visual_weight must be >= 0, got {visual_weight}")
        if max_word_repetitions < 1:
            raise ValueError(
                f"max_word_repetitions must be >= 1, got {max_word_repetitions}"
            )
        if retriever is None or model is None or training_captions is None:
            loaded = _load_default_artifacts()
            retriever = retriever or loaded["retriever"]
            model = model or loaded["model"]
            training_captions = (
                training_captions if training_captions is not None else loaded["training_captions"]
            )
        self.retriever = retriever
        self.model = model
        self.training_captions = {
            image_id: [list(tokens) for tokens in sequences]
            for image_id, sequences in training_captions.items()
        }
        self.top_k = top_k
        self.max_length = max_length
        self.min_length = min_length
        self.visual_weight = visual_weight
        self.max_word_repetitions = max_word_repetitions

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def retrieve(self, image_path: str | Path, top_k: int | None = None) -> list[dict]:
        """Retrieve the Top-K visually similar training images (Phase 5)."""
        return self.retriever.retrieve(
            image_path, top_k=top_k if top_k is not None else self.top_k
        )

    def candidate_captions(
        self, retrieval_results: Sequence[Mapping]
    ) -> list[list[str]]:
        """Training captions of the retrieved images (the visual evidence)."""
        return collect_candidate_captions(retrieval_results, self.training_captions)

    def _decode(self, candidate_captions: Sequence[Sequence[str]]) -> dict:
        """Greedy decoding over the candidate words (see module docstring)."""
        return greedy_decode(
            self.model,
            candidate_captions,
            max_length=self.max_length,
            min_length=self.min_length,
            visual_weight=self.visual_weight,
            max_word_repetitions=self.max_word_repetitions,
        )

    def model_context(self, tokens: Sequence[str]) -> tuple[str, ...]:
        """The last ``n-1`` generated tokens, for the Phase 7 model."""
        return tuple(tokens[-(self.model.n - 1) :]) if self.model.n > 1 else ()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_from_retrieval_results(
        self, retrieval_results: Sequence[Mapping]
    ) -> dict:
        """Generate a caption from already-computed Phase 5 results."""
        candidate_captions = self.candidate_captions(retrieval_results)
        decoded = self._decode(candidate_captions)
        return {
            "retrieved": [dict(result) for result in retrieval_results],
            "candidate_caption_count": decoded["candidate_caption_count"],
            "candidate_word_count": decoded["candidate_word_count"],
            "tokens": decoded["tokens"],
            "caption": decoded["caption"],
        }

    def generate(self, image_path: str | Path, top_k: int | None = None) -> dict:
        """Full pipeline for one image: retrieval → candidates → decoding."""
        retrieval_results = self.retrieve(image_path, top_k=top_k)
        result = self.generate_from_retrieval_results(retrieval_results)
        result["query"] = Path(image_path).name
        return result

    def generate_caption(self, image_path: str | Path, top_k: int | None = None) -> str:
        """Convenience wrapper returning only the caption string."""
        return self.generate(image_path, top_k=top_k)["caption"]


def _load_default_artifacts() -> dict:
    """Load the Phase 4/5/6/7 artifacts from the configured locations."""
    config = load_config()
    fusion_dir = resolve_path(config["paths"]["features_dir"]) / FUSION_SUBDIR
    captions_dir = resolve_path(config["paths"]["captions_dir"])
    language_model_dir = resolve_path(config["paths"]["language_model_dir"])
    vocabulary_tokens, _size = load_phase6_vocabulary(captions_dir / "vocabulary.json")
    return {
        "retriever": ImageRetriever(fusion_dir),
        "model": load_model(
            language_model_dir / "ngram_model.json", vocabulary_tokens=vocabulary_tokens
        ),
        "training_captions": load_training_captions(captions_dir / "train_captions.csv"),
    }
