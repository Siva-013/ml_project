"""Phase 8 — command-line caption generation for one image.

Executes the complete classical pipeline: preprocessing → classical
features → fusion/PCA → retrieval → candidate captions → greedy trigram
decoding.

Run from the project root::

    python -m src.caption_generation.run --image 1000268201_693b08cb0e.jpg
    python -m src.caption_generation.run --image D:/path/to/photo.jpg --top-k 5 --max-length 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.caption_generation.generator import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MIN_LENGTH,
    DEFAULT_TOP_K,
    CaptionGenerator,
)
from src.config import load_config
from src.retrieval.run_retrieval import resolve_query_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.caption_generation.run",
        description="Generate a caption for one image using classical visual "
        "retrieval + the Phase 7 trigram language model (greedy decoding).",
    )
    parser.add_argument("--image", required=True, help="Query image path or dataset filename")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"Retrieved neighbours (default: {DEFAULT_TOP_K})")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH,
                        help=f"Maximum caption words (default: {DEFAULT_MAX_LENGTH})")
    parser.add_argument("--min-length", type=int, default=DEFAULT_MIN_LENGTH,
                        help=f"Minimum caption words (default: {DEFAULT_MIN_LENGTH})")
    parser.add_argument("--visual-weight", type=float, default=1.0,
                        help="Weight of the retrieval-evidence bonus (default: 1.0)")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        query_path = resolve_query_path(args.image, config)
        generator = CaptionGenerator(
            top_k=args.top_k,
            max_length=args.max_length,
            min_length=args.min_length,
            visual_weight=args.visual_weight,
        )
        result = generator.generate(query_path)
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as error:
        print(f"[error] {error}", file=sys.stderr)
        return 1

    print("Caption Generation")
    print("------------------")
    print()
    print(f"Query:")
    print(result["query"])
    print()
    print(f"Retrieved images (candidate source: training captions only):")
    for rank, retrieved in enumerate(result["retrieved"], start=1):
        print(
            f"{rank}. {retrieved['image_id']}    "
            f"similarity={retrieved['similarity']:.4f}"
        )
    print()
    print(
        f"Candidate captions: {result['candidate_caption_count']} "
        f"({result['candidate_word_count']} candidate words)"
    )
    print()
    print("Generated caption:")
    print(result["caption"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
