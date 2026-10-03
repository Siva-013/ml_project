"""Phase 7 — statistical n-gram language model (Laplace smoothing)."""

from src.language_model.ngram import (
    NGramLanguageModel,
    load_model,
    save_model,
)
from src.language_model.smoothing import LaplaceSmoothing

__all__ = ["NGramLanguageModel", "LaplaceSmoothing", "load_model", "save_model"]
