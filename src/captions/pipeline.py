"""Phase 6 — caption preprocessing pipeline (vocabulary from train only).

Reads the validated Phase 1 ``captions.csv``, normalizes and tokenizes
every caption, wraps it in ``<START>``/``<END>``, builds the vocabulary
from **training captions only**, encodes all captions to integer ID
sequences (unknown words → ``<UNK>``) and writes the Phase 6 outputs:

    data/captions/
    ├── train_captions.csv         image_id, caption_id, split,
    ├── validation_captions.csv    raw_caption, clean_caption,
    └── test_captions.csv          tokens (JSON string, <START>…<END>)
    ├── vocabulary.json            token ↔ ID maps, frequencies, specials
    ├── encoded_captions.json      per split: {image_id, caption_id, token_ids}
    └── caption_metadata.json      counts, rules, provenance

The 91 unassigned images are excluded from every Phase 6 output and from
vocabulary construction (documented in the metadata); they remain in the
untouched Phase 1 ``captions.csv``. Phase 1 files are read, never modified.

Run from the project root::

    python -m src.captions.pipeline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.captions.preprocess import PUNCTUATION_CHARS, normalize_caption, preprocess_caption
from src.captions.vocabulary import SPECIAL_TOKEN_PURPOSES, Vocabulary
from src.config import load_config, resolve_path

#: Splits processed into language-model data.
LM_SPLITS: tuple[str, ...] = ("train", "validation", "test")

_OUTPUT_FILES = (
    "train_captions.csv",
    "validation_captions.csv",
    "test_captions.csv",
    "vocabulary.json",
    "encoded_captions.json",
    "caption_metadata.json",
)

_REQUIRED_COLUMNS = {"image_id", "caption_id", "caption", "split"}


def _load_captions(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Phase 1 captions and separate LM splits from unassigned rows."""
    captions_path = processed_dir / "captions.csv"
    if not captions_path.exists():
        raise FileNotFoundError(
            f"{captions_path} not found — run 'python -m src.dataset' first"
        )
    captions = pd.read_csv(captions_path)
    missing = _REQUIRED_COLUMNS - set(captions.columns)
    if missing:
        raise ValueError(f"{captions_path} is missing columns: {sorted(missing)}")

    lm_rows = captions[captions["split"].isin(LM_SPLITS)].reset_index(drop=True)
    unassigned = captions[captions["split"] == "unassigned"]
    if lm_rows.empty:
        raise ValueError("No train/validation/test captions found")
    return lm_rows, unassigned


def run_caption_preprocessing(
    processed_dir: str | Path,
    output_dir: str | Path,
    *,
    min_frequency: int = 1,
) -> dict:
    """Run the complete Phase 6 pipeline and persist every artifact.

    Returns:
        The metadata dictionary that is also written to
        ``caption_metadata.json``.
    """
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    lm_rows, unassigned = _load_captions(processed_dir)

    # -- preprocess every caption (deterministic: CSV order) ------------
    token_lists: list[list[str]] = []
    clean_captions: list[str] = []
    for row in lm_rows.itertuples(index=False):
        try:
            token_lists.append(preprocess_caption(row.caption))
            clean_captions.append(normalize_caption(row.caption))
        except ValueError as error:
            raise ValueError(f"caption {row.image_id}#{row.caption_id}: {error}") from error
    lm_rows = lm_rows.assign(tokens=token_lists, clean_caption=clean_captions)

    # -- vocabulary: TRAINING CAPTIONS ONLY ------------------------------
    train_token_lists = lm_rows.loc[lm_rows["split"] == "train", "tokens"]
    vocabulary = Vocabulary.build(train_token_lists, min_frequency=min_frequency)

    # -- encode -----------------------------------------------------------
    encoded_ids = [vocabulary.encode(tokens) for tokens in lm_rows["tokens"]]

    # -- persist ----------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in LM_SPLITS:
        subset = lm_rows[lm_rows["split"] == split]
        frame = pd.DataFrame(
            {
                "image_id": subset["image_id"].to_numpy(),
                "caption_id": subset["caption_id"].to_numpy(),
                "split": subset["split"].to_numpy(),
                "raw_caption": subset["caption"].to_numpy(),
                "clean_caption": subset["clean_caption"].to_numpy(),
                # documented format: JSON array string of the full token
                # sequence, including the <START>/<END> boundary tokens
                "tokens": [json.dumps(tokens, ensure_ascii=False) for tokens in subset["tokens"]],
            }
        )
        frame.to_csv(output_dir / f"{split}_captions.csv", index=False)

    (output_dir / "vocabulary.json").write_text(
        json.dumps(vocabulary.to_json_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    sequences: dict[str, list[dict]] = {split: [] for split in LM_SPLITS}
    for (split, image_id, caption_id), ids in zip(
        lm_rows[["split", "image_id", "caption_id"]].itertuples(index=False, name=None),
        encoded_ids,
    ):
        sequences[split].append(
            {"image_id": image_id, "caption_id": int(caption_id), "token_ids": ids}
        )
    (output_dir / "encoded_captions.json").write_text(
        json.dumps(
            {
                "format": (
                    "per split, a list of {image_id, caption_id, token_ids}; "
                    "token_ids encode <START> tokens <END> using vocabulary.json; "
                    "out-of-vocabulary tokens are encoded as the <UNK> id"
                ),
                "sequences": sequences,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    per_image = lm_rows.groupby("image_id").size()
    metadata = {
        "dataset": "Flickr8k (project dataset; 1 documented upstream exclusion)",
        "total_images": int(lm_rows["image_id"].nunique()),
        "total_captions": int(len(lm_rows)),
        "train_images": int(lm_rows.loc[lm_rows["split"] == "train", "image_id"].nunique()),
        "validation_images": int(
            lm_rows.loc[lm_rows["split"] == "validation", "image_id"].nunique()
        ),
        "test_images": int(lm_rows.loc[lm_rows["split"] == "test", "image_id"].nunique()),
        "train_captions": int((lm_rows["split"] == "train").sum()),
        "validation_captions": int((lm_rows["split"] == "validation").sum()),
        "test_captions": int((lm_rows["split"] == "test").sum()),
        "unassigned_images": int(unassigned["image_id"].nunique()),
        "unassigned_captions": int(len(unassigned)),
        "captions_per_image": {
            "expected": 5,
            "minimum": int(per_image.min()),
            "maximum": int(per_image.max()),
        },
        "vocabulary_size": vocabulary.size,
        "special_tokens": dict(SPECIAL_TOKEN_PURPOSES),
        "minimum_frequency": int(min_frequency),
        "normalization_rules": [
            "lowercase",
            "ASCII punctuation characters replaced by spaces",
            "whitespace runs collapsed to a single space",
            "leading/trailing whitespace stripped",
            "no stemming, no lemmatization",
        ],
        "tokenization_method": "whitespace split (str.split) after normalization",
        "vocabulary_built_from": "training captions only",
        "unknown_token_policy": (
            "tokens absent from the training vocabulary are encoded as <UNK>; "
            "validation/test-only tokens never enter the vocabulary"
        ),
        "outputs": list(_OUTPUT_FILES),
    }
    (output_dir / "caption_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.captions.pipeline",
        description="Phase 6: normalize, tokenize and encode Flickr8k captions; "
        "build the vocabulary from training captions only.",
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--processed-dir",
        default=None,
        help="Phase 1 processed directory (default: paths.processed_dir)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Phase 6 output directory (default: paths.captions_dir)",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=1,
        help="Minimum training frequency for a vocabulary token (default: 1)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        processed_dir = (
            resolve_path(args.processed_dir)
            if args.processed_dir
            else resolve_path(config["paths"]["processed_dir"])
        )
        output_dir = (
            resolve_path(args.output_dir)
            if args.output_dir
            else resolve_path(config["paths"]["captions_dir"])
        )
        metadata = run_caption_preprocessing(
            processed_dir, output_dir, min_frequency=args.min_frequency
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"[error] {error}", file=sys.stderr)
        return 1

    print("Phase 6 caption preprocessing complete:")
    print()
    print(f"images processed    : {metadata['total_images']}")
    print(f"captions processed  : {metadata['total_captions']}")
    print(f"train captions      : {metadata['train_captions']}")
    print(f"validation captions : {metadata['validation_captions']}")
    print(f"test captions       : {metadata['test_captions']}")
    print(f"vocabulary size     : {metadata['vocabulary_size']}")
    print(f"vocabulary source   : {metadata['vocabulary_built_from']}")
    print(f"outputs             : {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
