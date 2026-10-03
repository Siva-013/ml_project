"""Phase 5 — run retrieval for one real Flickr8k image (demonstration CLI).

Example (from the project root)::

    python -m src.retrieval.run_retrieval --image 1000268201_693b08cb0e.jpg --top-k 5
    python -m src.retrieval.run_retrieval --image D:/path/to/photo.jpg --top-k 10

The ``--image`` value may be an explicit path or a bare dataset filename
(resolved against ``paths.images_dir`` from the config). The gallery is the
6,000-image training split of the Phase 4 PCA space; nothing is fitted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import load_config, resolve_path
from src.features.fusion import FUSION_SUBDIR
from src.retrieval.retriever import ImageRetriever


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.retrieval.run_retrieval",
        description="Retrieve the Top-K most visually similar training "
        "images for one query image.",
    )
    parser.add_argument(
        "--image", required=True, help="Query image path or dataset filename"
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--fusion-dir", default=None, help="Phase 4 output directory override"
    )
    parser.add_argument(
        "--include-self",
        action="store_true",
        help="Do not exclude a training query image from its own results",
    )
    return parser.parse_args(argv)


def resolve_query_path(image: str, config: dict) -> Path:
    """Accept either an explicit path or a dataset filename."""
    candidate = Path(image)
    if candidate.exists():
        return candidate
    dataset_path = resolve_path(config["paths"]["images_dir"]) / image
    if dataset_path.exists():
        return dataset_path
    raise FileNotFoundError(
        f"Query image not found: tried '{candidate}' and '{dataset_path}'"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        query_path = resolve_query_path(args.image, config)
        fusion_dir = (
            resolve_path(args.fusion_dir)
            if args.fusion_dir
            else resolve_path(config["paths"]["features_dir"]) / FUSION_SUBDIR
        )
        retriever = ImageRetriever(fusion_dir)
        results = retriever.retrieve(
            query_path, top_k=args.top_k, exclude_self=not args.include_self
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"[error] {error}", file=sys.stderr)
        return 1

    print("Image Retrieval")
    print("---------------")
    print(f"Query  : {query_path.name}")
    print(
        f"Gallery: {retriever.gallery_size} training images "
        f"({retriever.dimension}-D PCA space)\n"
    )
    print(f"Top {len(results)} results:")
    for rank, result in enumerate(results, start=1):
        print(f"{rank}. {result['image_id']}    similarity={result['similarity']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
