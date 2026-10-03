"""Phase 4 — diagnostic visualization of the fused PCA space (PC1 vs PC2).

Plots every train/validation/test image in the plane of the first two
principal components, coloured by split. This is purely for understanding
the learned feature space and for demonstrations; it is not used for
training, and no retrieval happens here.

Run from the project root::

    python -m src.features.visualize_pca [--output-dir DIR]

Requires the Phase 4 outputs (``python -m src.features.fusion``).
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

DEFAULT_OUTPUT_DIR = "outputs/visualizations"

_SPLIT_STYLES = (
    ("train", "tab:blue", 6, 0.35),
    ("validation", "tab:orange", 14, 0.75),
    ("test", "tab:green", 14, 0.75),
)


def build_pca_scatter(fusion_dir: Path, output_dir: Path) -> Path:
    """Render the PC1-vs-PC2 scatter and return the written path."""
    pca_figure_path = fusion_dir / "pca.joblib"
    if not pca_figure_path.exists():
        raise FileNotFoundError(
            f"{pca_figure_path} not found — run 'python -m src.features.fusion' first"
        )
    import joblib

    pca = joblib.load(pca_figure_path)
    ratio = pca.explained_variance_ratio_

    figure, axis = plt.subplots(figsize=(9, 7))
    for label, color, size, alpha in _SPLIT_STYLES:
        features = np.load(fusion_dir / f"{label}_features.npy")
        axis.scatter(
            features[:, 0],
            features[:, 1],
            s=size,
            alpha=alpha,
            color=color,
            label=f"{label} ({len(features)})",
            edgecolors="none",
        )
    axis.set_xlabel(f"PC1 ({ratio[0]:.1%} variance)")
    axis.set_ylabel(f"PC2 ({ratio[1]:.1%} variance)")
    total = ratio[0] + ratio[1]
    axis.set_title(
        "Phase 4 fused feature space — first two principal components "
        f"({total:.1%} of variance shown)"
    )
    axis.legend(loc="best")
    figure.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "pca_scatter_pc1_pc2.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.features.visualize_pca",
        description="Diagnostic scatter plot of the fused PCA space (PC1 vs PC2).",
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--fusion-dir",
        default=None,
        help="Phase 4 output directory (default: <features_dir>/fusion)",
    )
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
        features_dir = resolve_path(config["paths"]["features_dir"])
        fusion_dir = (
            resolve_path(args.fusion_dir) if args.fusion_dir else features_dir / FUSION_SUBDIR
        )
        output_path = build_pca_scatter(fusion_dir, resolve_path(args.output_dir))
    except FileNotFoundError as error:
        print(f"[error] {error}", file=sys.stderr)
        return 1
    print(f"PCA scatter written to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
