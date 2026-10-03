"""Phase 5 — visualise retrieval results for one query (diagnostic).

Renders the query image next to its Top-K retrieved gallery images, using
the actual Flickr8k files on disk. For professor demonstrations; not part
of the pipeline.

Run from the project root::

    python -m src.retrieval.visualize --image <image> --top-k 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering; no display required

import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config, resolve_path
from src.features.fusion import FUSION_SUBDIR
from src.preprocessing.image_preprocessing import TARGET_SIZE, preprocess_image
from src.retrieval.run_retrieval import resolve_query_path
from src.retrieval.retriever import ImageRetriever

DEFAULT_OUTPUT_DIR = "outputs/visualizations"


def build_retrieval_figure(
    query_path: Path, results: list[dict], images_dir: Path, output_dir: Path
) -> Path:
    """Render query + Top-K thumbnails and return the written path."""
    panels = len(results) + 1
    columns = min(panels, 6)
    rows = (panels + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(2.2 * columns, 2.6 * rows))
    axes = np.atleast_2d(axes)

    query = preprocess_image(query_path)
    axes.flat[0].imshow(query.rgb)
    axes.flat[0].set_title("Query", fontsize=9)
    for axis in list(axes.flat)[: len(results) + 1]:
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes.flat[len(results) + 1 :]:
        axis.axis("off")

    for index, result in enumerate(results, start=1):
        image = preprocess_image(images_dir / result["image_id"])
        axes.flat[index].imshow(image.rgb)
        axes.flat[index].set_title(
            f"{index}. {result['image_id']}\nsim={result['similarity']:.4f}", fontsize=7
        )

    figure.suptitle(
        f"Phase 5 retrieval — query '{query_path.name}' "
        f"(gallery: training images, cosine similarity)"
    )
    figure.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"retrieval_{query_path.stem}.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.retrieval.visualize",
        description="Render a query image next to its Top-K retrieved images.",
    )
    parser.add_argument("--image", required=True, help="Query image path or filename")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR, help=f"PNG directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        query_path = resolve_query_path(args.image, config)
        images_dir = resolve_path(config["paths"]["images_dir"])
        retriever = ImageRetriever(
            resolve_path(config["paths"]["features_dir"]) / FUSION_SUBDIR
        )
        results = retriever.retrieve(query_path, top_k=args.top_k)
        output_path = build_retrieval_figure(
            query_path, results, images_dir, resolve_path(args.output_dir)
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"[error] {error}", file=sys.stderr)
        return 1
    print(f"Retrieval figure written to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
