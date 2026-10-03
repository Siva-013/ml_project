"""Phase 7 — train the n-gram language model on Phase 6 training captions.

Pipeline (the model sees **training captions only**; the validation split
is only scored, and the test split is not touched at all — it stays
reserved for Phase 9):

    load Phase 6 vocabulary + training captions
      → fit NGramLanguageModel(n=3, Laplace α=1.0)
      → validation perplexity (primary model)
      → unigram/bigram comparison on the same data
      → save data/language_model/ngram_model.json + model_metadata.json

Run from the project root::

    python -m src.language_model.train
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from src.captions.vocabulary import END, PAD, START, UNK
from src.config import load_config, resolve_path
from src.language_model.evaluate import compare_orders, evaluate_perplexity
from src.language_model.ngram import NGramLanguageModel, save_model
from src.language_model.smoothing import LaplaceSmoothing

#: The primary model for this project.
PRIMARY_ORDER = 3
COMPARISON_ORDERS: tuple[int, ...] = (1, 2, 3)


def load_phase6_captions(csv_path: Path) -> list[list[str]]:
    """Load the token sequences written by Phase 6 (``<PAD>`` filtered)."""
    frame = pd.read_csv(csv_path)
    if "tokens" not in frame.columns:
        raise ValueError(f"{csv_path} has no 'tokens' column — run Phase 6 first")
    sequences = []
    for tokens_json in frame["tokens"]:
        tokens = json.loads(tokens_json)
        sequences.append([token for token in tokens if token != PAD])
    return sequences


def load_phase6_vocabulary(vocabulary_path: Path) -> tuple[set[str], int]:
    """Return (vocabulary token set, vocabulary size) from Phase 6."""
    data = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    token_to_id = data["token_to_id"]
    return set(token_to_id), int(data["vocabulary_size"])


def run_training(
    captions_dir: Path,
    output_dir: Path,
    *,
    order: int = PRIMARY_ORDER,
    alpha: float = 1.0,
    with_comparison: bool = True,
) -> dict:
    """Train on Phase 6 training captions, evaluate, persist, and return metadata."""
    vocabulary_path = captions_dir / "vocabulary.json"
    if not vocabulary_path.exists():
        raise FileNotFoundError(
            f"{vocabulary_path} not found — run 'python -m src.captions.pipeline' first"
        )
    vocabulary_tokens, vocabulary_size = load_phase6_vocabulary(vocabulary_path)
    train_captions = load_phase6_captions(captions_dir / "train_captions.csv")
    validation_captions = load_phase6_captions(captions_dir / "validation_captions.csv")

    start = time.perf_counter()
    model = NGramLanguageModel(
        n=order,
        smoothing=LaplaceSmoothing(alpha=alpha),
        vocabulary_tokens=vocabulary_tokens,
    ).fit(train_captions)
    validation_metrics = evaluate_perplexity(model, validation_captions)
    elapsed = time.perf_counter() - start

    comparison = (
        compare_orders(
            train_captions,
            validation_captions,
            orders=COMPARISON_ORDERS,
            alpha=alpha,
            vocabulary_tokens=vocabulary_tokens,
        )
        if with_comparison
        else {}
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_model(model, output_dir / "ngram_model.json")

    metadata = {
        "model_type": "NGramLanguageModel",
        "n": order,
        "smoothing": "laplace",
        "alpha": alpha,
        "vocabulary_size": vocabulary_size,
        "training_caption_count": len(train_captions),
        "training_only": True,
        "validation_caption_count": len(validation_captions),
        "special_tokens": {"PAD": PAD, "UNK": UNK, "START": START, "END": END},
        "training_token_count": model.total_tokens,
        "ngram_counts": {
            str(ngram_order): len(model.ngram_counts[ngram_order])
            for ngram_order in range(1, order + 1)
        },
        "validation": {
            "perplexity": validation_metrics["perplexity"],
            "prediction_count": validation_metrics["prediction_count"],
            "mean_log_probability": validation_metrics["mean_log_probability"],
        },
        "order_comparison_validation_perplexity": {
            str(order_): metrics["perplexity"] for order_, metrics in comparison.items()
        },
        "vocabulary_source": "data/captions/vocabulary.json (Phase 6, training only)",
        "runtime_seconds": round(elapsed, 2),
    }
    (output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.language_model.train",
        description="Phase 7: train the trigram language model on Phase 6 "
        "training captions and evaluate on the validation split.",
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--captions-dir", default=None, help="Phase 6 output directory override"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Language-model output directory (default: paths.language_model_dir)",
    )
    parser.add_argument(
        "--order", type=int, default=PRIMARY_ORDER, help="Model order (default: 3)"
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="Laplace smoothing alpha (default: 1.0)"
    )
    parser.add_argument(
        "--no-comparison",
        action="store_true",
        help="Skip the unigram/bigram comparison fits",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        captions_dir = (
            resolve_path(args.captions_dir)
            if args.captions_dir
            else resolve_path(config["paths"]["captions_dir"])
        )
        output_dir = (
            resolve_path(args.output_dir)
            if args.output_dir
            else resolve_path(config["paths"]["language_model_dir"])
        )
        metadata = run_training(
            captions_dir,
            output_dir,
            order=args.order,
            alpha=args.alpha,
            with_comparison=not args.no_comparison,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"[error] {error}", file=sys.stderr)
        return 1

    comparison = metadata.get("order_comparison_validation_perplexity", {})
    model_label = "Trigram" if metadata["n"] == 3 else f"{metadata['n']}-gram"
    print("Phase 7 N-gram language model training complete:")
    print()
    print(f"training captions    : {metadata['training_caption_count']}")
    print(f"validation captions  : {metadata['validation_caption_count']}")
    print(f"vocabulary size      : {metadata['vocabulary_size']}")
    print()
    print(f"model                : {model_label}")
    print(f"smoothing            : {metadata['smoothing'].capitalize()}")
    print(f"alpha                : {metadata['alpha']}")
    print()
    for label, key in (("unigrams", "1"), ("bigrams", "2"), ("trigrams", "3")):
        if key in metadata["ngram_counts"]:
            print(f"{label:<21}: {metadata['ngram_counts'][key]}")
    print()
    for order_key in sorted(comparison):
        label = {1: "unigram", 2: "bigram", 3: "trigram"}.get(
            int(order_key), f"{order_key}-gram"
        )
        print(f"validation PP ({label + ')':<12}: {comparison[order_key]:.4f}")
    print()
    print(f"outputs              : {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
