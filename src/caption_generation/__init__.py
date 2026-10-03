"""Phase 8 — caption generation (retrieval evidence + greedy trigram decoding)."""

from src.caption_generation.candidates import (
    build_candidate_words,
    collect_candidate_captions,
    load_training_captions,
)
from src.caption_generation.generator import CaptionGenerator

__all__ = [
    "CaptionGenerator",
    "build_candidate_words",
    "collect_candidate_captions",
    "load_training_captions",
]
