"""Phase 6 — vocabulary construction from training captions only.

The vocabulary maps tokens to deterministic integer IDs:

1. The four special tokens always occupy IDs 0-3, in the fixed order
   ``<PAD>, <UNK>, <START>, <END>``.
2. All other tokens are ordered by frequency (descending), with
   alphabetical order as the tie-breaker — fully deterministic, never
   dependent on dictionary or set iteration order.

Data-leakage rule: :meth:`Vocabulary.build` must be fed **training caption
tokens only**. Validation/test tokens that are absent from the training
vocabulary encode to the ``<UNK>`` ID and never enter the vocabulary.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

#: Special tokens, in their fixed ID order (0-3).
SPECIAL_TOKENS: tuple[str, ...] = ("<PAD>", "<UNK>", "<START>", "<END>")
PAD, UNK, START, END = SPECIAL_TOKENS

#: Purpose of each special token (used in metadata/documentation).
SPECIAL_TOKEN_PURPOSES: dict[str, str] = {
    PAD: "future padding to equalise sequence lengths (unused in Phase 6)",
    UNK: "token absent from the training vocabulary",
    START: "marks the beginning of a caption",
    END: "marks the end of a caption",
}


class Vocabulary:
    """Deterministic token ↔ ID mapping built from training tokens."""

    def __init__(
        self,
        token_to_id: dict[str, int],
        token_frequency: dict[str, int],
        min_frequency: int,
    ) -> None:
        self._token_to_id = dict(token_to_id)
        self._id_to_token = {index: token for token, index in self._token_to_id.items()}
        self._token_frequency = dict(token_frequency)
        self._min_frequency = min_frequency

    @classmethod
    def build(
        cls, token_sequences: Iterable[Iterable[str]], min_frequency: int = 1
    ) -> "Vocabulary":
        """Build the vocabulary from token sequences (training captions only).

        Args:
            token_sequences: Iterable of token lists (typically preprocessed
                captions including ``<START>``/``<END>``).
            min_frequency: Minimum training frequency for a token to enter
                the vocabulary. The default of 1 keeps the full training
                vocabulary; higher values drop rarer tokens (they would then
                encode to ``<UNK>``).

        Returns:
            A :class:`Vocabulary` with deterministic ID assignment.
        """
        if min_frequency < 1:
            raise ValueError(f"min_frequency must be >= 1, got {min_frequency}")

        counts: Counter[str] = Counter()
        for sequence in token_sequences:
            counts.update(sequence)

        ordered = sorted(
            (
                (token, count)
                for token, count in counts.items()
                if count >= min_frequency
            ),
            key=lambda item: (-item[1], item[0]),
        )
        token_to_id = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        for token, _count in ordered:
            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)
        token_frequency = {token: counts.get(token, 0) for token in token_to_id}
        return cls(token_to_id, token_frequency, min_frequency)

    # -- lookup ---------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of vocabulary entries (specials included)."""
        return len(self._token_to_id)

    @property
    def token_to_id_map(self) -> dict[str, int]:
        return dict(self._token_to_id)

    @property
    def id_to_token_map(self) -> dict[int, str]:
        return dict(self._id_to_token)

    @property
    def token_frequency(self) -> dict[str, int]:
        return dict(self._token_frequency)

    @property
    def min_frequency(self) -> int:
        return self._min_frequency

    @property
    def unk_id(self) -> int:
        return self._token_to_id[UNK]

    def contains(self, token: str) -> bool:
        """True if ``token`` is in the vocabulary."""
        return token in self._token_to_id

    def token_to_id(self, token: str) -> int:
        """Map a token to its ID; unknown tokens map to the ``<UNK>`` ID."""
        return self._token_to_id.get(token, self._token_to_id[UNK])

    def id_to_token(self, index: int) -> str:
        """Map an ID back to its token.

        Raises:
            KeyError: If the ID is not part of the vocabulary.
        """
        return self._id_to_token[int(index)]

    def encode(self, tokens: Iterable[str]) -> list[int]:
        """Encode a token sequence; unknown tokens become the ``<UNK>`` ID."""
        return [self.token_to_id(token) for token in tokens]

    def decode(self, ids: Iterable[int]) -> list[str]:
        """Decode an ID sequence back to tokens.

        Raises:
            KeyError: If an ID is not part of the vocabulary.
        """
        return [self._id_to_token[int(index)] for index in ids]

    # -- serialisation ----------------------------------------------------

    def to_json_dict(self) -> dict:
        """JSON-serialisable representation (deterministic ordering)."""
        return {
            "special_tokens": {"PAD": PAD, "UNK": UNK, "START": START, "END": END},
            "token_to_id": self._token_to_id,
            # JSON object keys must be strings
            "id_to_token": {
                str(index): token for index, token in self._id_to_token.items()
            },
            "token_frequency": self._token_frequency,
            "vocabulary_size": self.size,
            "built_from": "train",
        }


def load_vocabulary(path: str | Path) -> Vocabulary:
    """Load a vocabulary previously written via :meth:`to_json_dict`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    token_to_id = {token: int(index) for token, index in data["token_to_id"].items()}
    frequency = {token: int(count) for token, count in data["token_frequency"].items()}
    return Vocabulary(token_to_id, frequency, min_frequency=1)
