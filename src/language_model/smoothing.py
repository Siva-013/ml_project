"""Phase 7 — smoothing for the statistical n-gram language model.

Maximum-likelihood estimation assigns probability 0 to any n-gram that was
never seen in training, which would make whole captions impossible.
Laplace (add-alpha) smoothing fixes this by reserving uniform mass:

    P(w | context) = (count(context, w) + alpha)
                     / (count(context) + alpha * V)

where ``V`` is the vocabulary size and ``alpha`` the smoothing parameter
(1.0 = classic add-one/Laplace; larger alpha flattens the distribution
towards uniform).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LaplaceSmoothing:
    """Add-alpha (Laplace) smoothing over a fixed vocabulary."""

    alpha: float = 1.0

    def __post_init__(self) -> None:
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}")

    @property
    def name(self) -> str:
        return "laplace" if self.alpha == 1.0 else f"add-{self.alpha:g}"

    def probability(self, count: int, context_count: int, vocabulary_size: int) -> float:
        """Smoothed conditional probability for one (context, word) event.

        Args:
            count: Times ``context, word`` was seen in training.
            context_count: Times ``context`` was seen in training
                (total token count for the unigram case).
            vocabulary_size: ``V`` — the fixed Phase 6 vocabulary size.

        Raises:
            ValueError: If ``vocabulary_size`` is not positive.
        """
        if vocabulary_size <= 0:
            raise ValueError(f"vocabulary_size must be positive, got {vocabulary_size}")
        return (count + self.alpha) / (context_count + self.alpha * vocabulary_size)
