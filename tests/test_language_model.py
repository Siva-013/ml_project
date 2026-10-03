"""Phase 7 tests: statistical n-gram language model.

Unit tests use a tiny hand-computable corpus so every count and probability
can be verified exactly. Integration tests use real Phase 6 captions and
the saved Phase 7 model artifacts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.captions.vocabulary import END, PAD, START, UNK
from src.language_model.evaluate import compare_orders, evaluate_perplexity
from src.language_model.ngram import NGramLanguageModel, load_model, save_model
from src.language_model.smoothing import LaplaceSmoothing
from src.language_model.train import load_phase6_captions

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAPTIONS_DIR = PROJECT_ROOT / "data" / "captions"
LM_DIR = PROJECT_ROOT / "data" / "language_model"
PHASE6_AVAILABLE = (CAPTIONS_DIR / "train_captions.csv").exists()
LM_AVAILABLE = (LM_DIR / "ngram_model.json").exists()

VOCAB = {START, END, "a", "dog", "runs", "walks", "the"}  # V = 7
ALPHA = 1.0

# Hand-computable corpus (V = 7)
CORPUS = [
    [START, "a", "dog", "runs", END],
    [START, "a", "dog", "walks", END],
    [START, "the", "dog", "runs", END],
]
# counts: <START> 3, <END> 3, a 2, dog 3, runs 2, walks 1, the 1 → total 15
# V used in tests = 7 → unigram P(dog) = (3+1)/(15+7) = 2/11; bigram
# P(dog | a) = (2+1)/(2+7) = 1/3; trigram P(runs | a, dog) = (1+1)/(2+7) = 2/9
# (sentence 3 contains "the", not "a", so (a, dog, runs) occurs once)


@pytest.fixture
def trigram() -> NGramLanguageModel:
    return NGramLanguageModel(
        n=3, smoothing=LaplaceSmoothing(alpha=ALPHA), vocabulary_tokens=VOCAB
    ).fit(CORPUS)


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


def test_unigram_counts():
    model = NGramLanguageModel(n=1, vocabulary_tokens=VOCAB).fit(CORPUS)
    assert model.get_ngram_count([START]) == 3
    assert model.get_ngram_count(["dog"]) == 3
    assert model.get_ngram_count(["walks"]) == 1
    assert model.get_ngram_count(["cat"]) == 0
    assert model.total_tokens == 15


def test_bigram_counts():
    model = NGramLanguageModel(n=2, vocabulary_tokens=VOCAB).fit(CORPUS)
    assert model.get_ngram_count([START, "a"]) == 2
    assert model.get_ngram_count(["a", "dog"]) == 2
    assert model.get_ngram_count(["dog", "runs"]) == 2
    assert model.get_ngram_count(["runs", END]) == 2
    assert model.get_ngram_count([START, "the"]) == 1
    assert model.get_ngram_count(["dog", "cat"]) == 0


def test_trigram_counts():
    model = NGramLanguageModel(n=3, vocabulary_tokens=VOCAB).fit(CORPUS)
    assert model.get_ngram_count([START, "a", "dog"]) == 2
    assert model.get_ngram_count(["a", "dog", "runs"]) == 1
    assert model.get_ngram_count(["dog", "runs", END]) == 2
    assert model.get_ngram_count([START, "the", "dog"]) == 1
    assert model.get_ngram_count(["the", "dog", "runs"]) == 1
    assert model.get_ngram_count(["a", "dog", "cat"]) == 0


def test_context_counts():
    model = NGramLanguageModel(n=3, vocabulary_tokens=VOCAB).fit(CORPUS)
    assert model.get_context_count(()) == 15  # unigram case: total tokens
    assert model.get_context_count([START]) == 3
    assert model.get_context_count(["a", "dog"]) == 2
    assert model.get_context_count(["a", "cat"]) == 0


def test_pad_never_contributes_to_counts():
    model = NGramLanguageModel(n=3, vocabulary_tokens=VOCAB).fit(
        [[START, "a", PAD, "dog", END]]
    )
    assert model.get_ngram_count(["a", PAD]) == 0
    assert model.get_ngram_count([PAD, "dog"]) == 0
    assert model.get_ngram_count(["a", "dog"]) == 1  # PAD filtered before counting
    assert model.total_tokens == 4


def test_ngram_count_rejects_too_long_ngram(trigram: NGramLanguageModel):
    with pytest.raises(ValueError, match="exceeds"):
        trigram.get_ngram_count([START, "a", "dog", "runs"])


# ---------------------------------------------------------------------------
# Probabilities (hand-computed with alpha=1, V=7)
# ---------------------------------------------------------------------------


def test_unigram_probability():
    model = NGramLanguageModel(n=1, vocabulary_tokens=VOCAB).fit(CORPUS)
    assert model.get_probability((), "dog") == pytest.approx(4 / 22)
    assert model.get_probability((), "cat") == pytest.approx(1 / 22)  # unseen → smoothed


def test_bigram_probability():
    model = NGramLanguageModel(n=2, vocabulary_tokens=VOCAB).fit(CORPUS)
    assert model.get_probability([START], "a") == pytest.approx(3 / 10)
    assert model.get_probability(["a"], "dog") == pytest.approx(3 / 9)
    assert model.get_probability(["a"], "cat") == pytest.approx(1 / 9)  # unseen word


def test_trigram_probability():
    model = NGramLanguageModel(n=3, vocabulary_tokens=VOCAB).fit(CORPUS)
    assert model.get_probability(["a", "dog"], "runs") == pytest.approx(2 / 9)
    assert model.get_probability(["a", "dog"], "cat") == pytest.approx(1 / 9)  # unseen word
    assert model.get_probability([START, "a"], "dog") == pytest.approx(3 / 9)


def test_probability_always_in_valid_range(trigram: NGramLanguageModel):
    for context, word in [
        ([], "dog"),
        ([START], "a"),
        (["a", "dog"], "runs"),
        (["a", "dog"], "cat"),
        (["x", "y"], "zebra"),
    ]:
        probability = trigram.get_probability(context, word)
        assert 0.0 < probability <= 1.0


def test_laplace_gives_unseen_ngrams_positive_probability(trigram: NGramLanguageModel):
    assert trigram.get_probability(["the", "cat"], "flies") > 0.0
    assert trigram.get_log_probability(["the", "cat"], "flies") > -math.inf


def test_higher_alpha_flattens_towards_uniform():
    weak = NGramLanguageModel(
        n=2, smoothing=LaplaceSmoothing(alpha=0.1), vocabulary_tokens=VOCAB
    ).fit(CORPUS)
    strong = NGramLanguageModel(
        n=2, smoothing=LaplaceSmoothing(alpha=10.0), vocabulary_tokens=VOCAB
    ).fit(CORPUS)
    # a seen bigram: more smoothing pulls its probability down towards 1/V
    assert strong.get_probability(["a"], "dog") < weak.get_probability(["a"], "dog")
    # an unseen bigram: more smoothing pushes its probability up towards 1/V
    assert strong.get_probability(["a"], "cat") > weak.get_probability(["a"], "cat")
    assert weak.smoothing.name == "add-0.1"
    assert LaplaceSmoothing(1.0).name == "laplace"


def test_long_context_is_truncated_to_model_order(trigram: NGramLanguageModel):
    # a 4-token context must be truncated to its last 2 tokens
    assert trigram.get_probability(["<x>", "the", "a", "dog"], "runs") == (
        trigram.get_probability(["a", "dog"], "runs")
    )


def test_smoothing_rejects_negative_alpha():
    with pytest.raises(ValueError, match="alpha"):
        LaplaceSmoothing(alpha=-1.0)


# ---------------------------------------------------------------------------
# Special tokens and unknown words
# ---------------------------------------------------------------------------


def test_start_token_conditions_first_word(trigram: NGramLanguageModel):
    # short context right after <START>: lower-order counts apply
    assert trigram.get_probability([START], "a") == pytest.approx(3 / 10)


def test_end_token_is_predicted(trigram: NGramLanguageModel):
    assert trigram.get_probability(["dog", "runs"], END) == pytest.approx(3 / 9)


def test_unk_is_a_valid_token_and_counts_as_unseen(trigram: NGramLanguageModel):
    assert trigram.get_ngram_count([UNK]) == 0
    probability = trigram.get_probability((), UNK)
    assert probability == pytest.approx(1 / 22)  # never seen in training
    assert 0.0 < probability <= 1.0


def test_unknown_words_map_to_unk_without_expanding_vocabulary(
    trigram: NGramLanguageModel,
):
    vocabulary_before = set(trigram.vocabulary_tokens)
    probability = trigram.get_probability(["a"], "zebra")
    assert probability == trigram.get_probability(["a"], UNK)
    assert probability > 0.0
    assert trigram.vocabulary_tokens == vocabulary_before  # unchanged
    assert not trigram.vocabulary_tokens.__contains__("zebra")


# ---------------------------------------------------------------------------
# Sentence probability and log probability
# ---------------------------------------------------------------------------


def test_sentence_probability_matches_hand_computation(
    trigram: NGramLanguageModel,
):
    # P(a|<START>) * P(dog|<START>,a) * P(runs|a,dog) * P(<END>|dog,runs)
    # = (3/10) * (3/9) * (2/9) * (3/9)
    expected = (3 / 10) * (3 / 9) * (2 / 9) * (3 / 9)
    sentence = [START, "a", "dog", "runs", END]
    assert trigram.sentence_probability(sentence) == pytest.approx(expected)
    assert trigram.sentence_log_probability(sentence) == pytest.approx(math.log(expected))


def test_sentence_log_probability_is_finite_and_negative(
    trigram: NGramLanguageModel,
):
    log_probability = trigram.sentence_log_probability(
        [START, "the", "dog", "walks", END]
    )
    assert math.isfinite(log_probability)
    assert log_probability < 0.0
    # probability reconstructs exactly from the log value
    assert trigram.sentence_probability([START, "the", "dog", "walks", END]) == (
        pytest.approx(math.exp(log_probability))
    )


def test_sentence_without_start_token_raises(trigram: NGramLanguageModel):
    with pytest.raises(ValueError, match="start"):
        trigram.sentence_log_probability(["a", "dog", END])


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------


def test_perplexity_positive_and_finite(trigram: NGramLanguageModel):
    perplexity = trigram.calculate_perplexity(CORPUS)
    assert perplexity > 0.0
    assert math.isfinite(perplexity)


def test_perplexity_matches_definition(trigram: NGramLanguageModel):
    # PP = exp(-1/N * sum log P) with N = total predicted tokens = 3 * 4 = 12
    perplexity = trigram.calculate_perplexity(CORPUS)
    total_log = sum(trigram.sentence_log_probability(s) for s in CORPUS)
    assert perplexity == pytest.approx(math.exp(-total_log / 12))


def test_seen_captions_perplexity_lower_than_novel_captions(
    trigram: NGramLanguageModel,
):
    seen = trigram.calculate_perplexity(CORPUS)
    novel = trigram.calculate_perplexity([[START, "the", "cat", "sleeps", END]])
    assert seen < novel  # the model scores its own training data better


def test_perplexity_empty_captions_raise(trigram: NGramLanguageModel):
    with pytest.raises(ValueError, match="zero predictions"):
        trigram.calculate_perplexity([])


# ---------------------------------------------------------------------------
# Determinism and training-only behaviour
# ---------------------------------------------------------------------------


def test_training_is_deterministic():
    first = NGramLanguageModel(n=3, vocabulary_tokens=VOCAB).fit(CORPUS)
    second = NGramLanguageModel(n=3, vocabulary_tokens=VOCAB).fit(CORPUS)
    assert first.ngram_counts.keys() == second.ngram_counts.keys()
    for order in first.ngram_counts:
        assert first.ngram_counts[order] == second.ngram_counts[order]
    assert first.get_probability(["a"], "dog") == second.get_probability(["a"], "dog")


def test_validation_only_word_never_enters_model():
    """Explicit leakage canary: 'zebra' appears only in the validation set."""
    model = NGramLanguageModel(n=3, vocabulary_tokens=VOCAB).fit(CORPUS)
    validation_captions = [[START, "a", "zebra", "gallops", END]]
    # fitting was already done; scoring validation must not create counts
    model.calculate_perplexity(validation_captions)
    assert model.get_ngram_count(["zebra"]) == 0
    assert model.get_ngram_count(["a", "zebra"]) == 0
    assert model.get_ngram_count(["a", "zebra", "gallops"]) == 0
    # the word maps through <UNK> at scoring time
    assert model.get_probability(["a"], "zebra") == model.get_probability(["a"], UNK)


def test_refit_resets_counts_training_starts_from_scratch():
    model = NGramLanguageModel(n=2, vocabulary_tokens=VOCAB)
    model.fit(CORPUS)
    count_after_first = model.get_ngram_count(["a", "dog"])
    model.fit(CORPUS)
    assert model.get_ngram_count(["a", "dog"]) == count_after_first  # reset, not accumulate


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_model_json_round_trip(tmp_path: Path):
    model = NGramLanguageModel(
        n=3, smoothing=LaplaceSmoothing(alpha=ALPHA), vocabulary_tokens=VOCAB
    ).fit(CORPUS)
    path = save_model(model, tmp_path / "ngram_model.json")
    reloaded = load_model(path, vocabulary_tokens=VOCAB)
    assert reloaded.n == 3
    assert reloaded.total_tokens == model.total_tokens
    for sentence in CORPUS:
        assert reloaded.sentence_log_probability(sentence) == pytest.approx(
            model.sentence_log_probability(sentence)
        )


# ---------------------------------------------------------------------------
# Integration: real Phase 6 captions and saved artifacts
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_data() -> tuple[list[list[str]], list[list[str]], set[str], int]:
    vocabulary_path = CAPTIONS_DIR / "vocabulary.json"
    vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    vocabulary_tokens = set(vocabulary["token_to_id"])
    train = load_phase6_captions(CAPTIONS_DIR / "train_captions.csv")
    validation = load_phase6_captions(CAPTIONS_DIR / "validation_captions.csv")
    return train, validation, vocabulary_tokens, vocabulary["vocabulary_size"]


@pytest.mark.skipif(
    not PHASE6_AVAILABLE, reason="data/captions not found — run Phase 6 first"
)
class TestRealCaptions:
    def test_real_caption_sentence_probability(self, real_data):
        train, _, vocabulary_tokens, _ = real_data
        model = NGramLanguageModel(
            n=3, smoothing=LaplaceSmoothing(alpha=ALPHA), vocabulary_tokens=vocabulary_tokens
        ).fit(train[:100])
        probability = model.sentence_probability(train[0])
        assert 0.0 < probability <= 1.0

    def test_real_model_learns_frequent_patterns(self, real_data):
        train, _, vocabulary_tokens, _ = real_data
        model = NGramLanguageModel(
            n=3, smoothing=LaplaceSmoothing(alpha=ALPHA), vocabulary_tokens=vocabulary_tokens
        ).fit(train)
        # "a man" and "a dog" are frequent Flickr8k phrases
        assert model.get_ngram_count(["a", "man"]) > 100
        assert model.get_ngram_count(["a", "dog"]) > 100


@pytest.mark.skipif(
    not LM_AVAILABLE,
    reason="data/language_model not found — run 'python -m src.language_model.train' first",
)
class TestSavedModel:
    def test_metadata_matches_spec(self):
        metadata = json.loads((LM_DIR / "model_metadata.json").read_text(encoding="utf-8"))
        assert metadata["model_type"] == "NGramLanguageModel"
        assert metadata["n"] == 3
        assert metadata["smoothing"] == "laplace"
        assert metadata["alpha"] == 1.0
        assert metadata["vocabulary_size"] == 7374
        assert metadata["training_caption_count"] == 30000
        assert metadata["training_only"] is True
        assert metadata["validation"]["perplexity"] > 0.0
        assert metadata["ngram_counts"]["1"] > 0
        assert metadata["ngram_counts"]["2"] > 0
        assert metadata["ngram_counts"]["3"] > 0

    def test_saved_model_round_trip_and_real_perplexity(self):
        vocabulary = json.loads(
            (CAPTIONS_DIR / "vocabulary.json").read_text(encoding="utf-8")
        )
        vocabulary_tokens = set(vocabulary["token_to_id"])
        model = load_model(LM_DIR / "ngram_model.json", vocabulary_tokens=vocabulary_tokens)
        assert model.n == 3
        assert model.total_tokens > 0
        validation = load_phase6_captions(CAPTIONS_DIR / "validation_captions.csv")
        perplexity = model.calculate_perplexity(validation[:100])
        assert math.isfinite(perplexity)
        assert perplexity > 0.0

    def test_order_comparison_prefers_lower_perplexity_information(
        self,
    ):
        metadata = json.loads((LM_DIR / "model_metadata.json").read_text(encoding="utf-8"))
        comparison = metadata["order_comparison_validation_perplexity"]
        assert set(comparison) == {"1", "2", "3"}
        assert all(value > 0.0 for value in comparison.values())
