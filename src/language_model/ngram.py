"""Phase 7 — generalized statistical n-gram language model.

One class covers unigram, bigram and trigram models:

    NGramLanguageModel(n=1)  →  P(w)
    NGramLanguageModel(n=2)  →  P(w_i | w_(i-1))
    NGramLanguageModel(n=3)  →  P(w_i | w_(i-2), w_(i-1))

Counting and estimation (maximum likelihood + Laplace smoothing):

    count_k(t_1..t_k)        — how often the k-token tuple occurred
    total_tokens             — number of fitted tokens
    P(w | context) with k = len(context):
        k = 0      : (count₁(w) + α)        / (total_tokens + αV)
        k = 1..n-1 : (count_{k+1}(ctx, w) + α) / (count_k(ctx) + αV)

Contexts are truncated to the last ``n-1`` tokens, so a longer context
silently degrades to the model's own order; a **shorter** context (only
possible at the start of a caption, right after ``<START>``) uses the
corresponding lower-order counts — exactly the scheme sketched by

    P(a | <START>) · P(dog | <START>, a) · P(runs | a, dog) · …

for ``<START> a dog runs <END>``.

Data-leakage contract: :meth:`fit` sees **training captions only**. The
vocabulary is the fixed Phase 6 vocabulary; tokens outside it (e.g. from
validation/test text) are mapped to ``<UNK>`` at probability time and never
expand the vocabulary or the counts. ``<PAD>`` never contributes to
counts — it is filtered out of every sequence.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

from src.captions.vocabulary import END, PAD, START, UNK
from src.language_model.smoothing import LaplaceSmoothing


class NGramLanguageModel:
    """Statistical n-gram model with Laplace smoothing over a fixed vocabulary."""

    def __init__(
        self,
        n: int,
        smoothing: LaplaceSmoothing | None = None,
        vocabulary_tokens: set[str] | None = None,
    ) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.n = n
        self.smoothing = smoothing or LaplaceSmoothing(alpha=1.0)
        self.vocabulary_tokens = (
            set(vocabulary_tokens) if vocabulary_tokens is not None else None
        )
        self.ngram_counts: dict[int, Counter[tuple[str, ...]]] = {
            order: Counter() for order in range(1, n + 1)
        }
        self.total_tokens = 0
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting (training captions only)
    # ------------------------------------------------------------------

    def fit(self, captions: Iterable[Sequence[str]]) -> "NGramLanguageModel":
        """Count n-grams over the training captions.

        Args:
            captions: Token sequences including ``<START>``/``<END>``
                (Phase 6 output). ``<PAD>`` tokens are removed before
                counting.

        Returns:
            ``self``, to allow chaining. Re-fitting resets all counts:
            training always starts from scratch, keeping the pipeline
            deterministic.
        """
        self.ngram_counts = {order: Counter() for order in range(1, self.n + 1)}
        self.total_tokens = 0
        for sequence in captions:
            tokens = [token for token in sequence if token != PAD]
            self.total_tokens += len(tokens)
            for order in range(1, self.n + 1):
                counter = self.ngram_counts[order]
                counter.update(
                    tuple(tokens[index : index + order])
                    for index in range(len(tokens) - order + 1)
                )
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Normalization and counting helpers
    # ------------------------------------------------------------------

    def _normalize_token(self, token: str) -> str:
        if self.vocabulary_tokens is not None and token not in self.vocabulary_tokens:
            return UNK
        return token

    def _normalize_context(self, context: Sequence[str]) -> tuple[str, ...]:
        """Map unknown tokens to <UNK>, truncate to the last n-1 tokens."""
        normalized = tuple(self._normalize_token(token) for token in context)
        return normalized[-(self.n - 1) :] if self.n > 1 else ()

    def get_ngram_count(self, ngram: Sequence[str]) -> int:
        """Training count of one n-gram tuple (0 if never seen)."""
        ngram = tuple(self._normalize_token(token) for token in ngram)
        if len(ngram) > self.n:
            raise ValueError(
                f"n-gram length {len(ngram)} exceeds model order {self.n}"
            )
        return int(self.ngram_counts[len(ngram)][ngram])

    def get_context_count(self, context: Sequence[str]) -> int:
        """Training count of one context (total token count for the unigram).

        The context is truncated to the last ``n-1`` tokens, matching
        :meth:`get_probability`.
        """
        normalized = self._normalize_context(context)
        if len(normalized) == 0:
            return self.total_tokens
        return int(self.ngram_counts[len(normalized)][normalized])

    # ------------------------------------------------------------------
    # Probabilities
    # ------------------------------------------------------------------

    def get_probability(self, context: Sequence[str], word: str) -> float:
        """Smoothed P(word | context).

        Short contexts (start of caption) use the corresponding lower-order
        counts; longer contexts are truncated to the last ``n-1`` tokens.
        Unknown tokens map to ``<UNK>``. Never raises for unseen n-grams:
        Laplace smoothing assigns them positive probability.
        """
        normalized_word = self._normalize_token(word)
        normalized_context = self._normalize_context(context)
        k = len(normalized_context)
        if k == 0:
            count = self.ngram_counts[1][(normalized_word,)]
            denominator = self.total_tokens
        else:
            count = self.ngram_counts[k + 1][normalized_context + (normalized_word,)]
            denominator = self.ngram_counts[k][normalized_context]
        return self.smoothing.probability(count, denominator, self._vocabulary_size())

    def get_log_probability(self, context: Sequence[str], word: str) -> float:
        """Natural-log probability; ``-inf`` for zero-probability events
        (only possible with ``alpha = 0`` on unseen n-grams)."""
        probability = self.get_probability(context, word)
        return math.log(probability) if probability > 0.0 else -math.inf

    def _vocabulary_size(self) -> int:
        if self.vocabulary_tokens is not None:
            return len(self.vocabulary_tokens)
        # no fixed vocabulary supplied: V = distinct fitted tokens
        return len(self.ngram_counts[1])

    # ------------------------------------------------------------------
    # Sentence scoring
    # ------------------------------------------------------------------

    def sentence_log_probability(self, tokens: Sequence[str]) -> float:
        """Log probability of a full caption (``<START>`` is context only).

        Each token except the leading ``<START>`` is predicted with a
        context of up to ``n-1`` preceding tokens.
        """
        if len(tokens) == 0 or tokens[0] != START:
            raise ValueError(
                f"Caption must start with {START!r} (Phase 6 format), got {tokens[:3]}"
            )
        total = 0.0
        for index in range(1, len(tokens)):
            word = tokens[index]
            context = tokens[max(0, index - (self.n - 1)) : index]
            total += self.get_log_probability(context, word)
        return total

    def sentence_probability(self, tokens: Sequence[str]) -> float:
        """Probability of a full caption (exp of the log probability)."""
        log_probability = self.sentence_log_probability(tokens)
        return math.exp(log_probability) if log_probability != -math.inf else 0.0

    # ------------------------------------------------------------------
    # Perplexity
    # ------------------------------------------------------------------

    def calculate_perplexity(self, captions: Iterable[Sequence[str]]) -> float:
        """Perplexity PP = exp( -(1/N) * Σ log P(w_i | context_i) ).

        ``N`` counts every *predicted* token across captions (all tokens
        except each caption's leading ``<START>``). ``<END>`` and ``<UNK>``
        count as ordinary predictions; ``<PAD>`` is filtered out.

        Raises:
            ValueError: If the captions contain no predictions.
        """
        total_log_probability = 0.0
        predictions = 0
        for sequence in captions:
            tokens = [token for token in sequence if token != PAD]
            if len(tokens) <= 1:
                continue
            total_log_probability += self.sentence_log_probability(tokens)
            predictions += len(tokens) - 1
        if predictions == 0:
            raise ValueError("Cannot compute perplexity over zero predictions")
        return math.exp(-total_log_probability / predictions)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_json_dict(self) -> dict:
        """JSON-serialisable model (count keys are space-joined tokens)."""
        return {
            "model_type": "NGramLanguageModel",
            "n": self.n,
            "smoothing": self.smoothing.name,
            "alpha": self.smoothing.alpha,
            "total_tokens": self.total_tokens,
            "counts": {
                str(order): {
                    " ".join(ngram): count
                    for ngram, count in sorted(counter.items())
                }
                for order, counter in self.ngram_counts.items()
            },
        }

    @classmethod
    def from_json_dict(
        cls,
        data: dict,
        vocabulary_tokens: set[str] | None = None,
    ) -> "NGramLanguageModel":
        """Rebuild a model from :meth:`to_json_dict` output."""
        model = cls(
            n=int(data["n"]),
            smoothing=LaplaceSmoothing(alpha=float(data["alpha"])),
            vocabulary_tokens=vocabulary_tokens,
        )
        for order_key, entries in data["counts"].items():
            counter = model.ngram_counts[int(order_key)]
            for ngram_key, count in entries.items():
                counter[tuple(ngram_key.split(" "))] = int(count)
        model.total_tokens = int(data["total_tokens"])
        model._fitted = True
        return model


def save_model(model: NGramLanguageModel, path: str | Path) -> Path:
    """Write the model to ``path`` as deterministic, inspectable JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(model.to_json_dict(), indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def load_model(
    path: str | Path, vocabulary_tokens: set[str] | None = None
) -> NGramLanguageModel:
    """Load a model previously written by :func:`save_model`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return NGramLanguageModel.from_json_dict(data, vocabulary_tokens=vocabulary_tokens)
