"""Phase 9 tests: BLEU/ROUGE metrics, retrieval proxies, leakage checks and
the experiment pipeline (small real sample for speed)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.captions.vocabulary import END, START
from src.evaluation.caption_metrics import (
    corpus_bleu,
    evaluate_captions,
    rouge_l,
    rouge_n,
    sentence_bleu,
    strip_boundary_tokens,
    unk_count,
)
from src.evaluation.error_analysis import classify_failure
from src.evaluation.experiment import run_split_experiment
from src.evaluation.retrieval_metrics import (
    containment,
    jaccard_overlap,
    retrieval_relevance,
)
from src.evaluation.run import main as evaluation_main
from src.config import load_config, resolve_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"
CAPTIONS_DIR = PROJECT_ROOT / "data" / "captions"
PHASE6_AVAILABLE = (CAPTIONS_DIR / "validation_captions.csv").exists()


# ---------------------------------------------------------------------------
# BLEU (hand-computed)
# ---------------------------------------------------------------------------


def test_bleu_perfect_match_scores_one():
    references = [[["a", "dog", "runs", "fast"]]]
    hypothesis = ["a", "dog", "runs", "fast"]
    scores = corpus_bleu(references, [hypothesis])
    assert scores[1] == pytest.approx(1.0)
    assert scores[4] == pytest.approx(1.0)


def test_bleu_short_sentence_has_no_higher_ngrams():
    # a 3-token sentence has no 4-grams, so p4 totals are 0 → BLEU-4 = 0
    scores = corpus_bleu([[["a", "dog", "runs"]]], [["a", "dog", "runs"]])
    assert scores[3] == pytest.approx(1.0)
    assert scores[4] == 0.0


def test_bleu_partial_overlap_hand_computed():
    # candidate: a dog walks  | reference: a dog runs
    # p1 = 2/3 (a, dog) ; p2 = 1/2 ((a,dog)) ;
    # cumulative BLEU-2 = (p1 * p2) ** 0.5
    scores = corpus_bleu([[["a", "dog", "runs"]]], [["a", "dog", "walks"]])
    assert scores[1] == pytest.approx(2 / 3)
    assert scores[2] == pytest.approx(math.sqrt(2 / 3 * 1 / 2))


def test_bleu_brevity_penalty_short_hypothesis():
    # candidate shorter than reference → BP = exp(1 - 3/2); unigram p1 = 1
    scores = corpus_bleu([[["a", "dog", "runs"]]], [["a", "dog"]])
    expected_bp = math.exp(1 - 3 / 2)
    assert scores[1] == pytest.approx(expected_bp)


def test_bleu_no_brevity_penalty_for_long_hypothesis():
    scores = corpus_bleu([[["a", "dog"]]], [["a", "dog", "runs"]])
    assert scores[1] == pytest.approx(2 / 3)  # pure precision, no BP


def test_bleu_multi_reference_clipping():
    # "a" appears twice in the hypothesis but at most once in either
    # reference → clipped to 1 → p1 = 2/3 (a, dog)
    references = [
        [["a", "dog"], ["a", "cat", "dog"]]
    ]
    scores = corpus_bleu(references, [["a", "a", "dog"]])
    assert scores[1] == pytest.approx(2 / 3)


def test_bleu_multi_reference_best_ngram():
    references = [[["the", "cat", "sits"], ["a", "dog", "runs"]]]
    scores = corpus_bleu(references, [["a", "dog", "runs"]])
    assert scores[3] == pytest.approx(1.0)


def test_bleu_disjoint_hypothesis_is_zero():
    scores = corpus_bleu([[["a", "dog"]]], [["x", "y"]])
    assert scores[1] == 0.0
    assert scores[4] == 0.0


def test_sentence_bleu_matches_corpus_bleu_on_single_example():
    sentence = sentence_bleu([["a", "dog", "runs"]], ["a", "dog", "walks"])
    corpus = corpus_bleu([[["a", "dog", "runs"]]], [["a", "dog", "walks"]])
    assert sentence == corpus


# ---------------------------------------------------------------------------
# ROUGE (hand-computed)
# ---------------------------------------------------------------------------


def test_rouge_1_perfect_and_partial():
    assert rouge_n(["a", "dog", "runs"], [["a", "dog", "runs"]], 1) == 1.0
    # candidate {a, dog, walks} vs reference {a, dog, runs}: overlap 2,
    # P = 2/3, R = 2/3 → F1 = 2/3
    assert rouge_n(["a", "dog", "walks"], [["a", "dog", "runs"]], 1) == pytest.approx(2 / 3)


def test_rouge_2_overlap():
    # bigrams of candidate: (a,dog) only in common → P = R = 1/2 → F1 = 1/2
    assert rouge_n(["a", "dog", "walks"], [["a", "dog", "runs"]], 2) == pytest.approx(0.5)


def test_rouge_l_lcs_hand_computed():
    # LCS = 3; P = 3/3 = 1, R = 3/6 = 0.5 → F1 = 2·1·0.5/1.5 = 2/3
    assert rouge_l(["the", "cat", "sat"], [["the", "cat", "sat", "on", "the", "mat"]]) == (
        pytest.approx(2 / 3)
    )
    # LCS("a dog runs", "a dog walks") = 2 → P = R = 2/3 → F1 = 2/3
    assert rouge_l(["a", "dog", "runs"], [["a", "dog", "walks"]]) == pytest.approx(2 / 3)


def test_rouge_takes_best_reference():
    references = [["x", "y", "z"], ["a", "dog", "runs"]]
    assert rouge_n(["a", "dog", "runs"], references, 1) == 1.0
    assert rouge_l(["a", "dog", "runs"], references) == 1.0


# ---------------------------------------------------------------------------
# Special tokens, empty captions, UNK handling
# ---------------------------------------------------------------------------


def test_boundary_tokens_stripped_before_scoring():
    tokens = strip_boundary_tokens([START, "a", "dog", END])
    assert tokens == ["a", "dog"]


def test_metrics_ignore_boundary_tokens():
    plain = ["a", "dog", "runs"]
    decorated = [START, "a", "dog", "runs", END]
    scores_plain = evaluate_captions([[plain]], [plain])
    scores_decorated = evaluate_captions([[plain]], [decorated])
    assert scores_plain["bleu_1"] == scores_decorated["bleu_1"]
    assert scores_plain["rouge_l"] == scores_decorated["rouge_l"]


def test_unk_counted_and_reported():
    assert unk_count(["a", "<UNK>", "dog"]) == 1
    scores = evaluate_captions([[["a", "dog"]]], [["<UNK>", "dog"]])
    # <UNK> is treated as an ordinary (unmatched) token, documented
    assert scores["bleu_1"] == pytest.approx(1 / 2)
    assert scores["unk_count"] == 1


def test_empty_hypothesis_scores_zero_without_crashing():
    scores = corpus_bleu([[["a", "dog"]]], [[]])
    assert scores[1] == 0.0
    assert rouge_n([], [["a", "dog"]], 1) == 0.0
    assert rouge_l([], [["a", "dog"]]) == 0.0


# ---------------------------------------------------------------------------
# Retrieval proxies
# ---------------------------------------------------------------------------


def test_retrieval_overlap_proxies():
    query = [["a", "dog", "in", "the", "grass"]]
    neighbour = [["a", "dog", "on", "the", "beach"]]
    proxies = retrieval_relevance(query, neighbour)
    # query tokens {a,dog,in,the,grass}, neighbour tokens {a,dog,on,the,beach}
    assert proxies["jaccard_overlap"] == pytest.approx(3 / 7)
    assert proxies["containment"] == pytest.approx(3 / 5)


def test_jaccard_and_containment_edge_cases():
    assert jaccard_overlap(set(), set()) == 0.0
    assert containment(set(), {"a"}) == 0.0
    assert jaccard_overlap({"a"}, {"a"}) == 1.0


# ---------------------------------------------------------------------------
# Failure classification heuristics
# ---------------------------------------------------------------------------


def test_failure_classification_categories():
    assert classify_failure("a man in a field", 0.0, 0.5) == "generic"
    assert "very_short" in classify_failure("a dog", 0.0, 0.5)
    assert classify_failure("a dog runs in the grass", 0.0, 0.5) == "none"
    assert "low_caption_overlap" in classify_failure("a dog runs", 0.5, 0.1)


# ---------------------------------------------------------------------------
# Real pipeline (small sample for speed)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def validation_sample(tmp_path_factory) -> dict:
    output_dir = tmp_path_factory.mktemp("evaluation")
    exit_code = evaluation_main(
        [
            "--split", "validation",
            "--all-models",
            "--max-images", "3",
            "--output-dir", str(output_dir),
        ]
    )
    assert exit_code == 0
    return {"output_dir": output_dir}


@pytest.mark.skipif(
    not PHASE6_AVAILABLE, reason="data/captions not found — run Phase 6 first"
)
class TestExperimentPipeline:
    def test_predictions_csv_structure(self, validation_sample):
        predictions = pd_read(validation_sample["output_dir"] / "validation_predictions.csv")
        assert list(predictions.columns) == [
            "image_id", "generated_caption",
            "reference_1", "reference_2", "reference_3", "reference_4", "reference_5",
            "top_k", "language_model", "caption_length", "unk_count",
        ]
        assert len(predictions) == 3
        assert (predictions["top_k"] == 5).all()
        assert (predictions["language_model"] == "trigram").all()
        assert (predictions["caption_length"] <= 20).all()

    def test_metrics_outputs(self, validation_sample):
        metrics = json.loads(
            (validation_sample["output_dir"] / "validation_metrics.json").read_text(encoding="utf-8")
        )
        assert set(metrics["model_metrics"]) == {"unigram", "bigram", "trigram"}
        for name, panel in metrics["model_metrics"].items():
            for key in ("bleu_1", "bleu_2", "bleu_3", "bleu_4", "rouge_1", "rouge_2", "rouge_l"):
                assert 0.0 <= panel[key] <= 1.0, (name, key)
        comparison = pd_read(validation_sample["output_dir"] / "language_model_comparison.csv")
        assert sorted(comparison["model"]) == ["bigram", "trigram", "unigram"]
        top_k = pd_read(validation_sample["output_dir"] / "top_k_comparison.csv")
        assert sorted(top_k["top_k"]) == [1, 3, 5, 10]
        retrieval = pd_read(validation_sample["output_dir"] / "retrieval_results.csv")
        assert set(retrieval["rank"].unique()) <= set(range(1, 11))  # K = 10 max

    def test_leakage_checks_pass(self, validation_sample):
        summary = json.loads(
            (validation_sample["output_dir"] / "evaluation_summary.json").read_text(encoding="utf-8")
        )
        assert summary["leakage_checks"] == {
            "training_only_vocabulary": True,
            "candidate_captions_training_only": True,
            "scaler_fitted_on_train_only": True,
            "pca_fitted_on_train_only": True,
            "splits_separated": True,
        }

    def test_qualitative_examples_written(self, validation_sample):
        qualitative = pd_read(validation_sample["output_dir"] / "qualitative_examples.csv")
        assert set(qualitative["quality"]) == {"good", "moderate", "poor"}
        assert "analysis" in qualitative.columns


def pd_read(path: Path):
    import pandas as pd

    return pd.read_csv(path)


def test_evaluation_is_deterministic(tmp_path: Path):
    """Same inputs → identical metrics (documented, no random sampling)."""
    if not PHASE6_AVAILABLE:
        pytest.skip("data/captions not found")
    config = load_config()
    output_a = tmp_path / "a"
    output_b = tmp_path / "b"
    for output in (output_a, output_b):
        assert evaluation_main(
            ["--split", "validation", "--max-images", "2", "--output-dir", str(output)]
        ) == 0
    assert (output_a / "validation_metrics.json").read_text(encoding="utf-8") == (
        output_b / "validation_metrics.json"
    ).read_text(encoding="utf-8")
