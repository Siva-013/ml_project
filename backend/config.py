"""Phase 10 — backend configuration and request-validation limits.

Everything here guards the API surface: allowed upload formats, size
limits, and the legal ranges for generation parameters. Origins are
restricted to the local React development server by default and can be
overridden with the ``ALLOWED_ORIGINS`` environment variable
(comma-separated). This is a development-oriented setting — a real
deployment would lock origins to its own domain.
"""

from __future__ import annotations

import os

#: Upload constraints.
MAX_UPLOAD_MB = 10
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})
ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset(
    {"image/jpeg", "image/jpg", "image/png"}
)

#: Generation-parameter constraints (the backend validates every value).
ALLOWED_MODELS: tuple[str, ...] = ("unigram", "bigram", "trigram")
DEFAULT_MODEL = "trigram"
MIN_TOP_K, MAX_TOP_K = 1, 10
DEFAULT_TOP_K = 5
MIN_MAX_LENGTH, MAX_MAX_LENGTH = 1, 30
DEFAULT_MAX_LENGTH = 20
MIN_MIN_LENGTH, MAX_MIN_LENGTH = 0, 10
DEFAULT_MIN_LENGTH = 3

#: CORS (local development only; override with ALLOWED_ORIGINS env var).
_DEFAULT_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"
ALLOWED_ORIGINS: list[str] = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",")
    if origin.strip()
]


def validate_parameters(
    top_k: int, model: str, max_length: int, min_length: int
) -> None:
    """Raise ``ValueError`` with a user-facing message for any bad parameter."""
    if not MIN_TOP_K <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between {MIN_TOP_K} and {MAX_TOP_K}")
    if model not in ALLOWED_MODELS:
        raise ValueError(f"model must be one of {list(ALLOWED_MODELS)}")
    if not MIN_MAX_LENGTH <= max_length <= MAX_MAX_LENGTH:
        raise ValueError(f"max_length must be between {MIN_MAX_LENGTH} and {MAX_MAX_LENGTH}")
    if not MIN_MIN_LENGTH <= min_length <= min(max_length, MAX_MIN_LENGTH):
        raise ValueError(
            f"min_length must be between {MIN_MIN_LENGTH} and "
            f"{min(max_length, MAX_MIN_LENGTH)}"
        )
