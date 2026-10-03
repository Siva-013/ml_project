"""Phase 6 — deterministic caption text normalization and tokenization.

Pipeline (all steps deterministic, no statistical or learned component):

```
raw caption
  → lowercase
  → punctuation replaced by spaces
  → whitespace collapsed + stripped
  → whitespace tokenization
  → [<START>] + tokens + [<END>]
```

**Punctuation policy** (documented and pinned by tests): every ASCII
punctuation character (``string.punctuation``) is replaced by a space and
whitespace is then collapsed. Flickr8k captions already carry
space-separated punctuation ("... an entry way ."), so this cleanly drops
sentence-final periods; it also splits contractions ("dog's" → "dog s")
and hyphenated compounds ("ten-year-old" → "ten year old") — accepted,
deterministic trade-offs that keep words from being glued together.
No stemming or lemmatization is applied: natural caption wording is
preserved for later generation.
"""

from __future__ import annotations

import re
import string

from src.captions.vocabulary import END, START

#: ASCII punctuation replaced by spaces during normalization.
PUNCTUATION_CHARS: str = string.punctuation

_PUNCTUATION_TABLE = str.maketrans({char: " " for char in PUNCTUATION_CHARS})
_WHITESPACE = re.compile(r"\s+")


def normalize_caption(text: object) -> str:
    """Normalize one caption: lowercase, punctuation → spaces, collapse gaps.

    Args:
        text: The raw caption string.

    Returns:
        The normalized caption (no leading/trailing whitespace).

    Raises:
        ValueError: If ``text`` is None or NaN (a missing caption is a data
            integrity problem, never silently an empty caption).
    """
    if text is None or (isinstance(text, float) and text != text):  # NaN check
        raise ValueError("Caption is missing (None/NaN) — refusing to normalize")
    if not isinstance(text, str):
        text = str(text)
    lowered = text.lower()
    without_punctuation = lowered.translate(_PUNCTUATION_TABLE)
    return _WHITESPACE.sub(" ", without_punctuation).strip()


def tokenize_caption(text: object) -> list[str]:
    """Tokenize a caption into words (whitespace split after normalization)."""
    return normalize_caption(text).split()


def add_boundary_tokens(tokens: list[str]) -> list[str]:
    """Wrap a token list in the ``<START>`` / ``<END>`` boundary tokens."""
    return [START, *tokens, END]


def preprocess_caption(text: object) -> list[str]:
    """Full per-caption preprocessing: normalize → tokenize → boundaries.

    Raises:
        ValueError: If the caption is missing or contains no tokens after
            normalization (e.g. punctuation-only text) — such captions
            cannot support the language model and are treated as data
            integrity problems, never silently dropped.
    """
    tokens = tokenize_caption(text)
    if not tokens:
        raise ValueError(
            f"Caption contains no tokens after normalization: {text!r}"
        )
    return add_boundary_tokens(tokens)
