"""Phase 3 — feature sanity visualisation for a single image (diagnostic).

Renders, for one dataset image: the resized RGB image, the grayscale input,
the HOG visualisation, the LBP code map and the per-channel HSV histograms.
Intended for understanding the features and for professor demonstrations;
it is not part of the extraction pipeline and uses no deep learning.

Run from the project root::

    python -m src.features.visualize <image_filename> [--output-dir DIR]

Example::

    python -m src.features.visualize 1000268201_693b08cb0e.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering; no display required

import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog as skimage_hog
from skimage.feature import local_binary_pattern

from src.config import load_config, resolve_path
from src.features.color_histogram import COLOR_BINS_PER_CHANNEL
from src.features.hog import (
    HOG_BLOCK_NORM,
    HOG_CELLS_PER_BLOCK,
    HOG_ORIENTATIONS,
    HOG_PIXELS_PER_CELL,
    HOG_TRANSFORM_SQRT,
    extract_hog,
    hog_feature_dimension,
)
from src.features.lbp import LBP_METHOD, LBP_POINTS, LBP_RADIUS, quantise_grayscale
from src.preprocessing.image_preprocessing import TARGET_SIZE, preprocess_image

DEFAULT_OUTPUT_DIR = "outputs/visualizations"

_CHANNEL_STYLES = (("H", "tab:orange"), ("S", "tab:green"), ("V", "tab:blue"))


def build_visualization(image_id: str, images_dir: Path, output_dir: Path) -> Path:
    """Render the sanity figure for one image and return the written path."""
    preprocessed = preprocess_image(images_dir / image_id, image_id=image_id)

    _, hog_image = skimage_hog(
        preprocessed.grayscale,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        transform_sqrt=HOG_TRANSFORM_SQRT,
        visualize=True,
        feature_vector=True,
    )
    lbp_map = local_binary_pattern(
        quantise_grayscale(preprocessed.grayscale),
        P=LBP_POINTS,
        R=LBP_RADIUS,
        method=LBP_METHOD,
    )
    hog_features = extract_hog(preprocessed.grayscale)

    figure, axes = plt.subplots(2, 3, figsize=(13.5, 8))
    axes[0, 0].imshow(preprocessed.rgb)
    axes[0, 0].set_title(f"Resized RGB {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
    axes[0, 1].imshow(preprocessed.grayscale, cmap="gray")
    axes[0, 1].set_title("Grayscale (Rec. 709 luma)")
    axes[0, 2].imshow(hog_image, cmap="gray")
    axes[0, 2].set_title("HOG visualization")
    axes[1, 0].imshow(lbp_map, cmap="gray", vmin=0, vmax=LBP_POINTS + 1)
    axes[1, 0].set_title(f"LBP codes (P={LBP_POINTS}, R={LBP_RADIUS}, uniform)")

    histogram_axes = axes[1, 1]
    for channel_index, (name, color) in enumerate(_CHANNEL_STYLES):
        counts, edges = np.histogram(
            preprocessed.hsv[:, :, channel_index],
            bins=COLOR_BINS_PER_CHANNEL,
            range=(0.0, 1.0),
        )
        centers = (edges[:-1] + edges[1:]) / 2
        histogram_axes.plot(
            centers, counts, drawstyle="steps-mid", color=color, label=name
        )
    histogram_axes.set_title(f"HSV histograms ({COLOR_BINS_PER_CHANNEL} bins/channel)")
    histogram_axes.set_xlabel("bin centre")
    histogram_axes.set_ylabel("pixel count")
    histogram_axes.legend()

    text_axes = axes[1, 2]
    text_axes.axis("off")
    text_axes.text(
        0.0,
        1.0,
        "Feature dimensions\n"
        f"HOG           : {hog_feature_dimension(TARGET_SIZE[::-1])}\n"
        f"Color hist.   : {COLOR_BINS_PER_CHANNEL} x 3 = 96 (L1)\n"
        f"LBP histogram : {LBP_POINTS} + 2 = 10 (L1)\n"
        "\nParameters\n"
        f"HOG: {HOG_ORIENTATIONS} orient, "
        f"{HOG_PIXELS_PER_CELL[0]}x{HOG_PIXELS_PER_CELL[1]} px cells, "
        f"{HOG_CELLS_PER_BLOCK[0]}x{HOG_CELLS_PER_BLOCK[1]} cell blocks, "
        f"{HOG_BLOCK_NORM}\n"
        f"LBP: P={LBP_POINTS}, R={LBP_RADIUS}, method={LBP_METHOD}\n"
        f"\nHOG vector norm: {np.linalg.norm(hog_features):.3f}",
        va="top",
        fontsize=10,
        family="monospace",
    )

    figure.suptitle(f"Phase 3 feature sanity check — {image_id}")
    figure.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"feature_sanity_{Path(image_id).stem}.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.features.visualize",
        description="Render a Phase 3 feature sanity figure for one image.",
    )
    parser.add_argument("image_id", help="Image filename inside paths.images_dir")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for the PNG (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        output_path = build_visualization(
            args.image_id,
            images_dir=resolve_path(config["paths"]["images_dir"]),
            output_dir=resolve_path(args.output_dir),
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"[error] {error}", file=sys.stderr)
        return 1
    print(f"Sanity figure written to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
