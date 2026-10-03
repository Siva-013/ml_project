"""Phase 6 tests: caption normalization, tokenization and vocabulary.

Unit tests cover the pure text-processing functions and the Vocabulary
class on small synthetic inputs. Integration tests run the real Phase 6
pipeline and verify its artifacts against the Phase 1 CSVs — including the
explicit leakage canary: every vocabulary token must come from training
captions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.captions.pipeline import LM_SPLITS, main, run_caption_preprocessing
from src.captions.preprocess import (
    add_boundary_tokens,
    normalize_caption,
    preprocess_caption,
    tokenize_caption,
)
from src.captions.vocabulary import (
    END,
    PAD,
    START,
    UNK,
    Vocabulary,
    load_vocabulary,
)
from src.config import load_config, resolve_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CAPTIONS_DIR = PROJECT_ROOT / "data" / "captions"
PHASE1_AVAILABLE = (PROCESSED_DIR / "captions.csv").exists()
PHASE6_AVAILABLE = (CAPTIONS_DIR / "caption_metadata.json").exists()


# ---------------------------------------------------------------------------
# Unit tests: normalization (spec tests 1-3)
# ---------------------------------------------------------------------------


def test_lowercasing():
    assert normalize_caption("DOG RUNS") == "dog runs"
    assert normalize_caption("A Dog Runs Through The Grass") == "a dog runs through the grass"


def test_whitespace_normalization():
    assert normalize_caption("a   dog    runs") == "a dog runs"
    assert normalize_caption("  tabs\tand\nnewlines  ") == "tabs and newlines"


def test_punctuation_policy():
    # documented policy: every ASCII punctuation char becomes a space,
    # whitespace is collapsed — pre-separated punctuation disappears,
    # contractions/hyphens split rather than glue
    assert normalize_caption("dog, running!") == "dog running"
    assert tokenize_caption("an entry way .") == ["an", "entry", "way"]
    assert tokenize_caption("ten-year-old child") == ["ten", "year", "old", "child"]
    assert tokenize_caption("the dog's leash") == ["the", "dog", "s", "leash"]


# ---------------------------------------------------------------------------
# Unit tests: tokenization and boundaries (spec tests 4-5)
# ---------------------------------------------------------------------------


def test_tokenization():
    assert tokenize_caption("a dog runs") == ["a", "dog", "runs"]
    assert tokenize_caption("A dog is running!") == ["a", "dog", "is", "running"]


def test_boundary_tokens_wrap_every_caption():
    assert add_boundary_tokens(["a", "dog"]) == [START, "a", "dog", END]
    tokens = preprocess_caption("A dog runs through the grass.")
    assert tokens[0] == START
    assert tokens[-1] == END
    assert tokens == [
        START, "a", "dog", "runs", "through", "the", "grass", END
    ]


def test_missing_caption_raises():
    with pytest.raises(ValueError, match="missing"):
        normalize_caption(None)
    with pytest.raises(ValueError, match="missing"):
        normalize_caption(float("nan"))


def test_empty_after_normalization_raises():
    with pytest.raises(ValueError, match="no tokens"):
        preprocess_caption("   ")
    with pytest.raises(ValueError, match="no tokens"):
        preprocess_caption("... !!")


# ---------------------------------------------------------------------------
# Unit tests: vocabulary (spec tests 6-10, 27 leakage canary)
# ---------------------------------------------------------------------------


def _train_sequences() -> list[list[str]]:
    return [
        [START, "a", "dog", "runs", END],
        [START, "a", "dog", "walks", END],
        [START, "the", "dog", "runs", END],
        [START, "a", "cat", "sleeps", END],
    ]


def test_special_tokens_exist_with_fixed_ids():
    vocab = Vocabulary.build(_train_sequences())
    assert vocab.contains(PAD) and vocab.contains(UNK)
    assert vocab.contains(START) and vocab.contains(END)
    assert vocab.token_to_id(PAD) == 0
    assert vocab.token_to_id(UNK) == 1
    assert vocab.token_to_id(START) == 2
    assert vocab.token_to_id(END) == 3


def test_vocabulary_is_deterministic():
    first = Vocabulary.build(_train_sequences())
    second = Vocabulary.build(_train_sequences())
    assert first.token_to_id_map == second.token_to_id_map
    assert first.token_frequency == second.token_frequency


def test_frequency_descending_then_alphabetical_ordering():
    # "ape" (2 occurrences) precedes "zoo" (1 occurrence — tie with nothing);
    # "dog" (3 occurrences) precedes both
    building = Vocabulary.build(
        [[START, "dog", "zoo", END], [START, "dog", "ape", END], [START, "dog", "ape", END]]
    )
    token_to_id = building.token_to_id_map
    assert token_to_id["dog"] < token_to_id["ape"] < token_to_id["zoo"]


def test_min_frequency_drops_rare_tokens():
    full = Vocabulary.build(_train_sequences(), min_frequency=1)
    rare = Vocabulary.build(_train_sequences(), min_frequency=2)
    assert full.contains("walks") and full.contains("sleeps")  # min_frequency=1 keeps all
    assert not rare.contains("walks")  # appears once → dropped
    assert rare.contains("dog") and rare.contains("runs")  # appear >= 2 times
    # dropped tokens encode to <UNK>
    assert rare.token_to_id("walks") == rare.unk_id


def test_unknown_word_maps_to_unk():
    vocab = Vocabulary.build(_train_sequences())
    assert vocab.token_to_id("horse") == vocab.unk_id
    assert vocab.encode([START, "a", "horse", "runs", END]) == [
        vocab.token_to_id(START),
        vocab.token_to_id("a"),
        vocab.unk_id,
        vocab.token_to_id("runs"),
        vocab.token_to_id(END),
    ]


def test_encode_decode_roundtrip_for_known_tokens():
    vocab = Vocabulary.build(_train_sequences())
    tokens = [START, "a", "dog", "runs", END]
    assert vocab.decode(vocab.encode(tokens)) == tokens


def test_decode_unknown_id_raises():
    vocab = Vocabulary.build(_train_sequences())
    with pytest.raises(KeyError):
        vocab.decode([vocab.size + 123])


def test_validation_only_word_never_enters_training_vocabulary():
    """Explicit leakage canary (spec section 27)."""
    train = [[START, "a", "dog", "runs", END]]
    vocab = Vocabulary.build(train)
    validation_caption = [START, "a", "zebra", "gallops", END]  # validation-only words
    encoded = vocab.encode(validation_caption)
    assert not vocab.contains("zebra")
    assert not vocab.contains("gallops")
    assert encoded == [vocab.token_to_id(START), vocab.token_to_id("a"), vocab.unk_id, vocab.unk_id, vocab.token_to_id(END)]


# ---------------------------------------------------------------------------
# Integration: real Phase 6 pipeline artifacts
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def caption_outputs() -> Path:
    """Run the real pipeline once if its outputs are missing."""
    config = load_config()
    output_dir = resolve_path(config["paths"]["captions_dir"])
    if not (output_dir / "caption_metadata.json").exists():
        run_caption_preprocessing(
            resolve_path(config["paths"]["processed_dir"]), output_dir
        )
    return output_dir


@pytest.fixture(scope="module")
def metadata(caption_outputs: Path) -> dict:
    return json.loads((caption_outputs / "caption_metadata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def phase1() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "captions.csv")


@pytest.mark.skipif(
    not PHASE1_AVAILABLE, reason="data/processed/captions.csv not found — run Phase 1"
)
class TestRealCaptionPipeline:
    def test_metadata_counts(self, metadata: dict):
        assert metadata["total_images"] == 8000
        assert metadata["total_captions"] == 40000
        assert metadata["train_images"] == 6000
        assert metadata["validation_images"] == 1000
        assert metadata["test_images"] == 1000
        assert metadata["train_captions"] == 30000
        assert metadata["validation_captions"] == 5000
        assert metadata["test_captions"] == 5000
        assert metadata["unassigned_images"] == 91
        assert metadata["captions_per_image"] == {
            "expected": 5,
            "minimum": 5,
            "maximum": 5,
        }
        assert metadata["vocabulary_built_from"] == "training captions only"

    def test_every_caption_has_boundary_tokens(self, caption_outputs: Path):
        for split in LM_SPLITS:
            frame = pd.read_csv(caption_outputs / f"{split}_captions.csv")
            tokens = frame["tokens"].apply(json.loads)
            assert all(seq[0] == START and seq[-1] == END for seq in tokens)

    def test_no_empty_processed_captions(self, caption_outputs: Path):
        for split in LM_SPLITS:
            frame = pd.read_csv(caption_outputs / f"{split}_captions.csv")
            tokens = frame["tokens"].apply(json.loads)
            assert all(len(seq) >= 3 for seq in tokens)  # <START> + words + <END>

    def test_image_caption_alignment_preserved(
        self, caption_outputs: Path, phase1: pd.DataFrame
    ):
        processed = pd.read_csv(caption_outputs / "train_captions.csv")
        merged = processed.merge(
            phase1[["image_id", "caption_id", "caption"]],
            on=["image_id", "caption_id"],
            how="left",
            validate="one_to_one",
        )
        assert merged["caption"].equals(merged["raw_caption"])
        # a known real caption survives normalization verbatim
        first = processed.iloc[0]
        assert first["clean_caption"] == normalize_caption(first["raw_caption"])

    def test_five_captions_per_image(self, caption_outputs: Path):
        for split in LM_SPLITS:
            frame = pd.read_csv(caption_outputs / f"{split}_captions.csv")
            assert (frame.groupby("image_id").size() == 5).all()

    def test_split_integrity(self, caption_outputs: Path, phase1: pd.DataFrame):
        for split in LM_SPLITS:
            frame = pd.read_csv(caption_outputs / f"{split}_captions.csv")
            assert (frame["split"] == split).all()
            expected_ids = set(phase1.loc[phase1["split"] == split, "image_id"])
            assert set(frame["image_id"]) == expected_ids

    def test_unassigned_images_excluded(self, caption_outputs: Path, phase1: pd.DataFrame):
        unassigned_ids = set(phase1.loc[phase1["split"] == "unassigned", "image_id"])
        for split in LM_SPLITS:
            frame = pd.read_csv(caption_outputs / f"{split}_captions.csv")
            assert not (set(frame["image_id"]) & unassigned_ids)
        metadata = json.loads(
            (caption_outputs / "caption_metadata.json").read_text(encoding="utf-8")
        )
        assert metadata["unassigned_images"] == 91

    def test_vocabulary_built_from_training_captions_only(
        self, caption_outputs: Path, phase1: pd.DataFrame
    ):
        """Explicit leakage verification on the real dataset (spec section 27)."""
        vocabulary = json.loads(
            (caption_outputs / "vocabulary.json").read_text(encoding="utf-8")
        )
        train_rows = phase1[phase1["split"] == "train"]
        train_tokens = {
            token
            for caption in train_rows["caption"]
            for token in tokenize_caption(caption)
        }
        vocabulary_words = set(vocabulary["token_to_id"]) - set(
            vocabulary["special_tokens"].values()
        )
        # every non-special vocabulary token must occur in the training captions
        assert vocabulary_words <= train_tokens

    def test_encoded_sequences_are_valid(
        self, caption_outputs: Path, phase1: pd.DataFrame
    ):
        encoded = json.loads(
            (caption_outputs / "encoded_captions.json").read_text(encoding="utf-8")
        )
        vocabulary = json.loads(
            (caption_outputs / "vocabulary.json").read_text(encoding="utf-8")
        )
        vocab_size = vocabulary["vocabulary_size"]
        for split in LM_SPLITS:
            for sequence in encoded["sequences"][split]:
                assert len(sequence["token_ids"]) >= 3
                assert all(0 <= int(tid) < vocab_size for tid in sequence["token_ids"])

    def test_real_caption_structure(self):
        tokens = preprocess_caption(
            "A child in a pink dress is climbing up a set of stairs in an entry way ."
        )
        assert tokens == [
            START, "a", "child", "in", "a", "pink", "dress", "is", "climbing",
            "up", "a", "set", "of", "stairs", "in", "an", "entry", "way", END,
        ]


@pytest.mark.skipif(
    not PHASE1_AVAILABLE, reason="data/processed/captions.csv not found — run Phase 1"
)
def test_pipeline_is_deterministic(tmp_path: Path):
    """Running the pipeline twice must produce identical vocabulary + encodings."""
    reference = CAPTIONS_DIR / "vocabulary.json"
    if not reference.exists():
        run_caption_preprocessing(PROCESSED_DIR, CAPTIONS_DIR)
    run_caption_preprocessing(PROCESSED_DIR, tmp_path / "run_a")
    run_caption_preprocessing(PROCESSED_DIR, tmp_path / "run_b")

    for filename in ("vocabulary.json", "encoded_captions.json"):
        assert (tmp_path / "run_a" / filename).read_text(encoding="utf-8") == (
            tmp_path / "run_b" / filename
        ).read_text(encoding="utf-8")
    # and identical to the reference run stored in data/captions
    assert (tmp_path / "run_a" / "vocabulary.json").read_text(encoding="utf-8") == (
        reference.read_text(encoding="utf-8")
    )


@pytest.mark.skipif(
    not PHASE1_AVAILABLE, reason="data/processed/captions.csv not found — run Phase 1"
)
def test_cli_main_smoke(tmp_path: Path):
    assert main(["--output-dir", str(tmp_path / "cli_run"), "--min-frequency", "1"]) == 0
    assert (tmp_path / "cli_run" / "vocabulary.json").exists()
    reloaded = load_vocabulary(tmp_path / "cli_run" / "vocabulary.json")
    assert reloaded.size > 0
