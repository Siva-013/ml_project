"""Phase 9 — evaluation CLI.

Runs the experiments on one split and writes every artifact under
``data/evaluation/``:

    python -m src.evaluation.run --split validation --all-models
    python -m src.evaluation.run --split validation --top-k 5
    python -m src.evaluation.run --split test          # final, after validation choices

Split discipline: validation is the experiment/selection split; the test
split is evaluated once, with the configuration already fixed from
validation, and its results must not feed back into any choice.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from src.captions.vocabulary import load_vocabulary
from src.caption_generation.candidates import load_training_captions
from src.config import load_config, resolve_path
from src.evaluation.comparison import model_comparison_frame, top_k_comparison_frame
from src.evaluation.error_analysis import build_qualitative_examples
from src.evaluation.experiment import (
    EVALUATION_TOP_K_VALUES,
    MODEL_ORDER,
    build_models,
    run_split_experiment,
)
from src.features.fusion import FUSION_SUBDIR
from src.language_model.train import load_phase6_captions

MODEL_NAMES = tuple(MODEL_ORDER)


def run_leakage_checks(captions_dir: Path) -> dict[str, bool]:
    """Automated data-leakage verification over the existing artifacts."""
    checks: dict[str, bool] = {}

    vocabulary = load_vocabulary(captions_dir / "vocabulary.json")
    from src.evaluation.experiment import load_training_token_sequences

    training_tokens = {
        token for sequence in load_training_token_sequences(captions_dir) for token in sequence
    }
    checks["training_only_vocabulary"] = (
        set(vocabulary.token_to_id_map) - {"<PAD>", "<UNK>", "<START>", "<END>"}
    ) <= training_tokens

    training_captions = load_training_captions(captions_dir / "train_captions.csv")
    phase1 = pd.read_csv(resolve_path(config_paths()["processed_dir"]) / "captions.csv")
    train_ids = set(phase1.loc[phase1["split"] == "train", "image_id"])
    non_train_ids = set(phase1.loc[phase1["split"] != "train", "image_id"])
    checks["candidate_captions_training_only"] = set(training_captions) <= train_ids and not (
        set(training_captions) & non_train_ids
    )

    paths = config_paths()
    fusion_dir = resolve_path(paths["features_dir"]) / FUSION_SUBDIR
    scalers_ok = True
    for filename in ("hog_scaler.joblib", "color_scaler.joblib", "lbp_scaler.joblib"):
        scaler = joblib.load(fusion_dir / "scalers" / filename)
        scalers_ok &= int(scaler.n_samples_seen_) == 6000
    checks["scaler_fitted_on_train_only"] = scalers_ok

    fusion_metadata = json.loads((fusion_dir / "fusion_metadata.json").read_text(encoding="utf-8"))
    checks["pca_fitted_on_train_only"] = (
        fusion_metadata["pca"]["fit_on"] == "train_only"
        and fusion_metadata["pca"]["train_samples"] == 6000
    )

    validation_metadata = json.loads(
        (captions_dir / "caption_metadata.json").read_text(encoding="utf-8")
    )
    checks["splits_separated"] = (
        validation_metadata["train_images"] == 6000
        and validation_metadata["validation_images"] == 1000
        and validation_metadata["test_images"] == 1000
        and validation_metadata["unassigned_images"] == 91
    )
    return checks


def config_paths() -> dict:
    return load_config()["paths"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluation.run",
        description="Phase 9: evaluate the caption-generation pipeline.",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help="Split to evaluate (validation = experiments; test = final run)",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_NAMES,
        default=None,
        help="Single language model to evaluate (default: trigram)",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Evaluate unigram, bigram and trigram (validation experiments)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Primary Top-K (default: 5)")
    parser.add_argument(
        "--max-images", type=int, default=None, help="Evaluate only the first N images"
    )
    parser.add_argument(
        "--output-dir", default=None, help="Evaluation output directory override"
    )
    return parser.parse_args(argv)


def selected_models(args: argparse.Namespace) -> list[str]:
    if args.all_models:
        return list(MODEL_NAMES)
    return [args.model] if args.model else ["trigram"]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        paths = config_paths()
        captions_dir = resolve_path(paths["captions_dir"])
        output_dir = (
            resolve_path(args.output_dir)
            if args.output_dir
            else resolve_path(paths.get("evaluation_dir", "data/evaluation"))
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        model_names = selected_models(args)
        models = build_models(captions_dir)
        models = {name: models[name] for name in model_names}
        experiment = run_split_experiment(
            args.split,
            models=models,
            top_k_values=EVALUATION_TOP_K_VALUES,
            primary_model=model_names[-1],
            primary_top_k=args.top_k,
            max_images=args.max_images,
        )

        prefix = args.split
        experiment["predictions"].to_csv(output_dir / f"{prefix}_predictions.csv", index=False)
        pd.DataFrame(experiment["retrieval_rows"]).to_csv(
            output_dir / "retrieval_results.csv", index=False
        )

        if args.split == "validation":
            perplexities = {
                name: evaluate_validation_perplexity(models[name], captions_dir)
                for name in models
            }
            comparison = model_comparison_frame(experiment["model_metrics"], perplexities)
            comparison.to_csv(output_dir / "language_model_comparison.csv", index=False)
            top_k_frame = top_k_comparison_frame(experiment["top_k_metrics"])
            top_k_frame.to_csv(output_dir / "top_k_comparison.csv", index=False)

            references_by_image = {
                image_id: refs
                for image_id, refs in zip(*_reference_groups(captions_dir, args.split))
            }
            retrieval_by_image: dict[str, list[dict]] = {}
            for row in experiment["retrieval_rows"]:
                if row["rank"] <= 5:
                    retrieval_by_image.setdefault(row["image_id"], []).append(
                        {"image_id": row["retrieved_image_id"], "similarity": row["similarity"]}
                    )
            jaccards = sorted(experiment["jaccard_by_image"].values())
            quartile = jaccards[len(jaccards) // 4] if jaccards else 0.0
            qualitative = build_qualitative_examples(
                experiment["predictions"],
                references_by_image,
                retrieval_by_image,
                jaccard_by_image=experiment["jaccard_by_image"],
                per_group=10,
                overlap_threshold=quartile,
            )
            qualitative.to_csv(output_dir / "qualitative_examples.csv", index=False)

            metrics_frame = comparison.rename(columns={"model": "configuration"})
            metrics_frame.to_csv(output_dir / f"{prefix}_metrics.csv", index=False)
            (output_dir / f"{prefix}_metrics.json").write_text(
                json.dumps(
                    {
                        "split": args.split,
                        "models": model_names,
                        "top_k_values": list(EVALUATION_TOP_K_VALUES),
                        "primary_metrics": experiment["primary_metrics"],
                        "model_metrics": experiment["model_metrics"],
                        "top_k_metrics": {
                            str(k): v for k, v in experiment["top_k_metrics"].items()
                        },
                        "retrieval_summary": {
                            str(k): v for k, v in experiment["retrieval_summary"].items()
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            metrics_frame = pd.DataFrame(
                [{"configuration": f"{model_names[-1]}_top_{args.top_k}",
                  **experiment["primary_metrics"]}]
            )
            metrics_frame.to_csv(output_dir / f"{prefix}_metrics.csv", index=False)
            (output_dir / f"{prefix}_metrics.json").write_text(
                json.dumps(
                    {
                        "split": args.split,
                        "model": model_names[-1],
                        "top_k": args.top_k,
                        "primary_metrics": experiment["primary_metrics"],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        leakage = run_leakage_checks(captions_dir)
        summary = {
            "split": args.split,
            "models": model_names,
            "primary_top_k": args.top_k,
            "metrics": experiment["primary_metrics"],
            "model_comparison_validation_perplexity": (
                {
                    name: evaluate_validation_perplexity(models[name], captions_dir)
                    for name in models
                }
                if args.split == "validation"
                else {}
            ),
            "top_k_comparison": {
                str(k): v for k, v in experiment["top_k_metrics"].items()
            },
            "retrieval_summary": {
                str(k): v for k, v in experiment["retrieval_summary"].items()
            },
            "leakage_checks": leakage,
        }
        (output_dir / "evaluation_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as error:
        print(f"[error] {error}", file=sys.stderr)
        return 1

    metrics = experiment["primary_metrics"]
    print("=" * 50)
    print(f"PHASE 9 — {args.split.upper()} EVALUATION COMPLETE")
    print("=" * 50)
    print(f"images evaluated     : {len(experiment['predictions'])}")
    print(f"models               : {', '.join(model_names)}")
    print(f"primary Top-K        : {args.top_k}")
    print(f"BLEU-1 / 2 / 3 / 4   : {metrics['bleu_1']:.4f} / {metrics['bleu_2']:.4f} "
          f"/ {metrics['bleu_3']:.4f} / {metrics['bleu_4']:.4f}")
    print(f"ROUGE-1 / 2 / L      : {metrics['rouge_1']:.4f} / {metrics['rouge_2']:.4f} "
          f"/ {metrics['rouge_l']:.4f}")
    print(f"avg caption length   : {metrics['average_caption_length']:.2f} words")
    print(f"<UNK> count          : {metrics['unk_count']}")
    if args.split == "validation":
        comparison = experiment["model_metrics"]
        for name in sorted(comparison):
            print(f"{name:<8} validation PP : {perplexities[name]:.2f}  "
                  f"BLEU-1 {comparison[name]['bleu_1']:.4f}  BLEU-4 {comparison[name]['bleu_4']:.4f}")
        for k in sorted(experiment["top_k_metrics"]):
            print(f"top_k={k:<3} BLEU-1 : {experiment['top_k_metrics'][k]['bleu_1']:.4f}  "
                  f"ROUGE-L : {experiment['top_k_metrics'][k]['rouge_l']:.4f}")
    print("leakage checks       : " + ", ".join(
        f"{name}={'PASS' if ok else 'FAIL'}" for name, ok in leakage.items()
    ))
    print(f"outputs              : {output_dir}")
    return 0


def _reference_groups(captions_dir: Path, split: str):
    from src.evaluation.experiment import reference_groups

    return reference_groups(load_split_captions_safe(captions_dir, split))


def load_split_captions_safe(captions_dir: Path, split: str) -> pd.DataFrame:
    from src.evaluation.experiment import load_split_captions

    return load_split_captions(captions_dir, split)


def evaluate_validation_perplexity(model, captions_dir: Path) -> float:
    from src.language_model.evaluate import evaluate_perplexity

    validation = load_phase6_captions(captions_dir / "validation_captions.csv")
    return evaluate_perplexity(model, validation)["perplexity"]


if __name__ == "__main__":
    sys.exit(main())
