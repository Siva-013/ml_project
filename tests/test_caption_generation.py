"""Phase 8 tests: caption generation (retrieval evidence + greedy trigram decoding).

Unit tests use a small synthetic training set with a fake retriever, so the
decoding rules (boundaries, lengths, repetition, determinism, training-only
candidates) are verified exactly. Integration tests run the full real
pipeline end to end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.captions.vocabulary import END, PAD, START
from src.caption_generation.candidates import (
    build_candidate_words,
    collect_candidate_captions,
    load_training_captions,
)
from src.caption_generation.generator import CaptionGenerator
from src.caption_generation.run import main as run_main
from src.config import load_config, resolve_path
from src.language_model.ngram import NGramLanguageModel
from src.language_model.smoothing import LaplaceSmoothing

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAPTIONS_DIR = PROJECT_ROOT / "data" / "captions"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PHASE6_AVAILABLE = (CAPTIONS_DIR / "train_captions.csv").exists()

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

SYNTHETIC_TRAIN_CAPTIONS: dict[str, list[list[str]]] = {
    "img_1.jpg": [
        [START, "a", "dog", "runs", "in", "the", "grass", END],
        [START, "a", "dog", "runs", "fast", END],
        [START, "the", "dog", "is", "in", "the", "grass", END],
        [START, "a", "dog", "in", "the", "grass", END],
        [START, "one", "dog", "runs", END],
    ],
    "img_2.jpg": [
        [START, "a", "dog", "plays", "in", "the", "grass", END],
        [START, "two", "dogs", "play", "in", "the", "grass", END],
        [START, "a", "dog", "runs", "in", "the", "grass", END],
        [START, "dogs", "run", "in", "the", "grass", END],
        [START, "a", "dog", "in", "the", "grass", END],
    ],
    "img_3.jpg": [
        [START, "a", "boat", "sails", "on", "the", "water", END],
        [START, "the", "boat", "is", "on", "the", "water", END],
        [START, "a", "red", "boat", "on", "the", "water", END],
        [START, "boats", "float", "on", "the", "water", END],
        [START, "a", "boat", "in", "the", "harbor", END],
    ],
}

SYNTHETIC_QUERY = "query.jpg"


class FakeRetriever:
    """Deterministic stand-in for the Phase 5 ImageRetriever."""

    def __init__(self, results: list[dict]):
        self._results = results

    def retrieve(self, image_path, top_k=None, exclude_self=True):
        return [dict(result) for result in self._results[: top_k or len(self._results)]]


def _dog_retrieval_results(top_k: int = 2) -> list[dict]:
    return [
        {"image_id": "img_1.jpg", "similarity": 0.9},
        {"image_id": "img_2.jpg", "similarity": 0.8},
        {"image_id": "img_3.jpg", "similarity": 0.7},
    ][:top_k]


def _synthetic_generator(**overrides) -> CaptionGenerator:
    model = NGramLanguageModel(
        n=3,
        smoothing=LaplaceSmoothing(alpha=1.0),
        vocabulary_tokens={
            token
            for tokens in SYNTHETIC_TRAIN_CAPTIONS.values()
            for sequence in tokens
            for token in sequence
        },
    ).fit([tokens for seqs in SYNTHETIC_TRAIN_CAPTIONS.values() for tokens in seqs])
    parameters = dict(
        retriever=FakeRetriever(_dog_retrieval_results()),
        model=model,
        training_captions=SYNTHETIC_TRAIN_CAPTIONS,
    )
    parameters.update(overrides)
    return CaptionGenerator(**parameters)


# ---------------------------------------------------------------------------
# Candidate extraction
# ---------------------------------------------------------------------------


def test_load_training_captions_groups_by_image(tmp_path: Path):
    import pandas as pd

    frame = pd.DataFrame(
        {
            "image_id": ["x.jpg", "x.jpg", "y.jpg"],
            "tokens": [
                json.dumps([START, "a", END]),
                json.dumps([START, "b", END]),
                json.dumps([START, "c", END]),
            ],
        }
    )
    csv_path = tmp_path / "train_captions.csv"
    frame.to_csv(csv_path, index=False)
    captions = load_training_captions(csv_path)
    assert list(captions) == ["x.jpg", "y.jpg"]
    assert captions["x.jpg"] == [[START, "a", END], [START, "b", END]]


def test_collect_candidate_captions_ranked_and_complete():
    candidates = collect_candidate_captions(
        _dog_retrieval_results(top_k=2), SYNTHETIC_TRAIN_CAPTIONS
    )
    assert len(candidates) == 10  # 2 images x 5 captions
    assert candidates[0] == SYNTHETIC_TRAIN_CAPTIONS["img_1.jpg"][0]
    assert candidates[5][1] == "a"  # first caption of img_2 begins at index 5


def test_collect_candidate_captions_rejects_non_training_image():
    results = [{"image_id": "validation_pic.jpg", "similarity": 0.9}]
    with pytest.raises(KeyError, match="training"):
        collect_candidate_captions(results, SYNTHETIC_TRAIN_CAPTIONS)


def test_candidate_words_deterministic_and_clean():
    words = build_candidate_words(
        collect_candidate_captions(_dog_retrieval_results(), SYNTHETIC_TRAIN_CAPTIONS)
    )
    assert START not in words and PAD not in words
    assert END in words  # choosing it is how the decoder stops
    # deterministic: frequency descending, ties alphabetical
    assert words == build_candidate_words(
        collect_candidate_captions(_dog_retrieval_results(), SYNTHETIC_TRAIN_CAPTIONS)
    )
    # "grass" appears in many candidate captions; "boat" belongs to img_3,
    # which was not retrieved — so it is correctly absent from the candidates
    assert "grass" in words
    assert "boat" not in words
    # retrieving img_3 as well makes "boat" a candidate
    words_with_img3 = build_candidate_words(
        collect_candidate_captions(_dog_retrieval_results(top_k=3), SYNTHETIC_TRAIN_CAPTIONS)
    )
    assert "boat" in words_with_img3


# ---------------------------------------------------------------------------
# Generation behaviour
# ---------------------------------------------------------------------------


@pytest.fixture
def generator() -> CaptionGenerator:
    return _synthetic_generator()


def test_generation_output_is_clean(generator: CaptionGenerator):
    result = generator.generate(SYNTHETIC_QUERY)
    caption = result["caption"]
    assert caption  # non-empty
    for banned in (START, PAD, END):
        assert banned not in result["tokens"]
        assert banned not in caption.split()


def test_generation_respects_max_length():
    generator = _synthetic_generator(max_length=4, min_length=1)
    caption = generator.generate_caption(SYNTHETIC_QUERY)
    assert 1 <= len(caption.split()) <= 4


def test_generation_respects_min_length():
    generator = _synthetic_generator(min_length=5, max_length=12)
    caption = generator.generate_caption(SYNTHETIC_QUERY)
    assert len(caption.split()) >= 5


def test_generation_is_deterministic(generator: CaptionGenerator):
    first = generator.generate(SYNTHETIC_QUERY)
    second = generator.generate(SYNTHETIC_QUERY)
    assert first["caption"] == second["caption"]
    assert first["tokens"] == second["tokens"]
    assert first["retrieved"] == second["retrieved"]


def test_no_immediate_repetition(generator: CaptionGenerator):
    for result in (
        generator.generate(SYNTHETIC_QUERY),
        _synthetic_generator(top_k=3).generate(SYNTHETIC_QUERY),
        _synthetic_generator(max_length=20, min_length=3).generate(SYNTHETIC_QUERY),
    ):
        tokens = result["tokens"]
        assert all(tokens[i] != tokens[i + 1] for i in range(len(tokens) - 1))


def test_max_word_repetitions_cap_enforced():
    generator = _synthetic_generator(max_word_repetitions=1, max_length=8)
    tokens = generator.generate(SYNTHETIC_QUERY)["tokens"]
    assert len(tokens) == len(set(tokens))  # every word used at most once


def test_output_words_come_from_candidate_captions_only(generator: CaptionGenerator):
    """Explicit training-only leakage verification (synthetic)."""
    result = generator.generate(SYNTHETIC_QUERY)
    retrieved_ids = [r["image_id"] for r in result["retrieved"]]
    candidate_words = {
        token
        for image_id in retrieved_ids
        for sequence in generator.training_captions[image_id]
        for token in sequence
    }
    # "boat"/"water" belong to img_3's captions, which were not retrieved
    assert "boat" not in result["caption"].split()
    assert set(result["tokens"]) <= candidate_words


def test_generate_from_retrieval_results_consumes_phase5_results(
    generator: CaptionGenerator,
):
    results = _dog_retrieval_results(top_k=3)
    result = generator.generate_from_retrieval_results(results)
    assert [r["image_id"] for r in result["retrieved"]] == [
        r["image_id"] for r in results
    ]
    assert result["candidate_caption_count"] == 15  # 3 images x 5 captions
    assert result["caption"]


def test_empty_candidate_pool_stops_deterministically():
    generator = _synthetic_generator(min_length=0, max_length=10)
    result = generator.generate_from_retrieval_results(
        [{"image_id": "img_1.jpg", "similarity": 1.0}]
    )
    # a candidate pool whose only generatable token is <END> stops immediately
    degenerate = CaptionGenerator(
        retriever=FakeRetriever([{"image_id": "img_1.jpg", "similarity": 1.0}]),
        model=generator.model,
        training_captions={"img_1.jpg": [[START, END]]},
        min_length=0,
    )
    assert degenerate.generate_caption("whatever.jpg") == ""


def test_invalid_generator_parameters_raise():
    with pytest.raises(ValueError, match="max_length"):
        _synthetic_generator(max_length=0)
    with pytest.raises(ValueError, match="min_length"):
        _synthetic_generator(max_length=3, min_length=5)
    with pytest.raises(ValueError, match="visual_weight"):
        _synthetic_generator(visual_weight=-1.0)
    with pytest.raises(ValueError, match="max_word_repetitions"):
        _synthetic_generator(max_word_repetitions=0)


def test_visual_weight_zero_pure_language_model(generator: CaptionGenerator):
    pure_lm = _synthetic_generator(visual_weight=0.0)
    result = pure_lm.generate(SYNTHETIC_QUERY)
    assert result["caption"]  # still works: pure n-gram greedy decoding
    assert set(result["tokens"]) <= {
        token
        for image_id in [r["image_id"] for r in result["retrieved"]]
        for sequence in generator.training_captions[image_id]
        for token in sequence
    }


# ---------------------------------------------------------------------------
# Integration: real artifacts and real images
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_generator() -> CaptionGenerator:
    return CaptionGenerator()


@pytest.mark.skipif(
    not PHASE6_AVAILABLE, reason="data/captions not found — run Phase 6 first"
)
class TestRealCaptionGeneration:
    def test_generator_initialization(self, real_generator: CaptionGenerator):
        assert real_generator.retriever.gallery_size == 6000
        assert real_generator.retriever.dimension == 1804
        assert real_generator.model.n == 3
        assert len(real_generator.training_captions) == 6000

    def test_training_only_caption_source(self, real_generator: CaptionGenerator):
        phase1 = pd_read_processed()
        train_ids = set(phase1.loc[phase1["split"] == "train", "image_id"])
        non_train = set(phase1.loc[phase1["split"] != "train", "image_id"])
        assert set(real_generator.training_captions) <= train_ids
        assert not (set(real_generator.training_captions) & non_train)

    def test_real_image_end_to_end(self, real_generator: CaptionGenerator):
        config = load_config()
        image_id = "1000268201_693b08cb0e.jpg"
        image_path = resolve_path(config["paths"]["images_dir"]) / image_id
        first = real_generator.generate(image_path)
        second = real_generator.generate(image_path)

        assert first["caption"]
        assert len(first["tokens"]) <= real_generator.max_length
        assert first == second  # deterministic
        assert [r["image_id"] for r in first["retrieved"]] != []
        assert all(
            r["image_id"] in real_generator.training_captions
            for r in first["retrieved"]
        )

    def test_real_cli_smoke(self, real_generator: CaptionGenerator, capsys):
        exit_code = run_main(["--image", "1000268201_693b08cb0e.jpg", "--top-k", "5"])
        assert exit_code == 0
        output = capsys.readouterr().out
        assert "Generated caption" in output
        assert "Retrieved images" in output


def pd_read_processed() -> "pd.DataFrame":
    import pandas as pd

    return pd.read_csv(PROCESSED_DIR / "captions.csv")
