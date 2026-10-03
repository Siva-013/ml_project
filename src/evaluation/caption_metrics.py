"""Phase 9 — caption quality metrics (BLEU and ROUGE), implemented from scratch.

All metrics operate on plain word tokens. The project's evaluation
tokenization is the Phase 6 normalization (lowercase, punctuation →
spaces, whitespace split): generated captions are split on spaces, and
reference captions use the Phase 6 ``clean_caption`` column split the same
way. Boundary tokens (``<START>``, ``<END>``, ``<PAD>``) are stripped
before scoring; ``<UNK>`` — which Phase 8 candidates cannot produce — is
kept as an ordinary token and reported separately via ``unk_count``.

**BLEU** (corpus-level, multi-reference, standard convention):
- n-gram precision pₙ: clipped hits over the whole corpus, where per
  sentence each n-gram count is clipped against the *maximum* count of
  that n-gram over the image's references;
- BLEU-k is the *cumulative* score: brevity penalty × geometric mean of
  p₁…p_k;
- brevity penalty uses the closest reference length (ties → shorter).

**ROUGE** (per-image, then corpus mean):
- ROUGE-n = max F1 (β = 1) of n-gram overlap between the hypothesis and
  any single reference (max over the five references);
- ROUGE-L = max F1 of LCS-based overlap over the references;
- the corpus score is the arithmetic mean of the per-image values
  (the averaging method, documented).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence

BOUNDARY_TOKENS: frozenset[str] = frozenset({"<START>", "<END>", "<PAD>"})


def strip_boundary_tokens(tokens: Sequence[str]) -> list[str]:
    """Remove boundary tokens (``<START>``/``<END>``/``<PAD>``) from a sequence."""
    return [token for token in tokens if token not in BOUNDARY_TOKENS]


def tokenize_for_metrics(text: str) -> list[str]:
    """Split an already-normalized caption into metric tokens."""
    return text.split()


def unk_count(tokens: Sequence[str]) -> int:
    """Occurrences of the ``<UNK>`` token (reported, never treated as a word)."""
    return sum(1 for token in tokens if token == "<UNK>")


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))


def _closest_reference_length(hypothesis_length: int, reference_lengths: Sequence[int]) -> int:
    return min(reference_lengths, key=lambda length: (abs(length - hypothesis_length), length))


def corpus_bleu(
    references: Sequence[Sequence[Sequence[str]]],
    hypotheses: Sequence[Sequence[str]],
    max_n: int = 4,
) -> dict[int, float]:
    """Corpus-level cumulative multi-reference BLEU-1 … BLEU-max_n.

    Args:
        references: One element per image; each element holds that image's
            reference token sequences (boundaries already stripped).
        hypotheses: One token sequence per image.

    Returns:
        ``{n: BLEU-n}`` — brevity penalty × geometric mean of p₁…pₙ.
    """
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    hits = [0] * (max_n + 1)
    totals = [0] * (max_n + 1)
    hypothesis_length = 0
    reference_length = 0

    for refs, hypothesis in zip(references, hypotheses):
        if not refs:
            continue
        hypothesis_length += len(hypothesis)
        reference_length += _closest_reference_length(
            len(hypothesis), [len(reference) for reference in refs]
        )
        for n in range(1, max_n + 1):
            hypothesis_ngrams = _ngrams(hypothesis, n)
            if not hypothesis_ngrams:
                continue
            max_reference_ngrams: Counter = Counter()
            for reference in refs:
                for ngram, count in _ngrams(reference, n).items():
                    max_reference_ngrams[ngram] = max(
                        max_reference_ngrams[ngram], count
                    )
            for ngram, count in hypothesis_ngrams.items():
                hits[n] += min(count, max_reference_ngrams[ngram])
                totals[n] += count

    precisions = [
        (hits[n] / totals[n]) if totals[n] > 0 else 0.0 for n in range(1, max_n + 1)
    ]
    if hypothesis_length == 0:
        brevity_penalty = 0.0
    elif hypothesis_length > reference_length:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1 - reference_length / hypothesis_length)

    scores: dict[int, float] = {}
    for n in range(1, max_n + 1):
        if brevity_penalty == 0.0 or any(p == 0.0 for p in precisions[:n]):
            scores[n] = 0.0
        else:
            log_mean = sum(math.log(p) for p in precisions[:n]) / n
            scores[n] = brevity_penalty * math.exp(log_mean)
    return scores


def sentence_bleu(
    references: Sequence[Sequence[str]], hypothesis: Sequence[str], max_n: int = 4
) -> dict[int, float]:
    """Sentence-level BLEU (same convention as :func:`corpus_bleu`, one image)."""
    return corpus_bleu([[list(refs) for refs in references]], [list(hypothesis)], max_n=max_n)


def _f1(overlap: int, candidate_total: int, reference_total: int) -> float:
    if overlap == 0 or candidate_total == 0 or reference_total == 0:
        return 0.0
    precision = overlap / candidate_total
    recall = overlap / reference_total
    return 2 * precision * recall / (precision + recall)


def rouge_n(
    candidate: Sequence[str],
    references: Sequence[Sequence[str]],
    n: int,
) -> float:
    """ROUGE-n F1: max over the references of the n-gram overlap F1."""
    candidate_ngrams = _ngrams(list(candidate), n)
    best = 0.0
    for reference in references:
        reference_ngrams = _ngrams(list(reference), n)
        if not candidate_ngrams or not reference_ngrams:
            continue
        overlap = sum(
            min(count, reference_ngrams[ngram])
            for ngram, count in candidate_ngrams.items()
            if ngram in reference_ngrams
        )
        best = max(
            best,
            _f1(overlap, sum(candidate_ngrams.values()), sum(reference_ngrams.values())),
        )
    return best


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        current = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous = current
    return previous[-1]


def rouge_l(candidate: Sequence[str], references: Sequence[Sequence[str]]) -> float:
    """ROUGE-L F1: max over the references of the LCS-based F1."""
    best = 0.0
    for reference in references:
        if not candidate or not reference:
            continue
        lcs = _lcs_length(candidate, reference)
        best = max(best, _f1(lcs, len(candidate), len(reference)))
    return best


def evaluate_captions(
    references: Sequence[Sequence[Sequence[str]]],
    hypotheses: Sequence[Sequence[str]],
) -> dict[str, float]:
    """The Phase 9 metric panel over a whole prediction set.

    Returns corpus BLEU-1..4, corpus ROUGE-1/2/L (per-image max-F1, mean
    over images), mean caption length, and total <UNK> count.
    """
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    stripped_hypotheses = [strip_boundary_tokens(h) for h in hypotheses]
    stripped_references = [
        [strip_boundary_tokens(reference) for reference in refs] for refs in references
    ]
    bleu = corpus_bleu(stripped_references, stripped_hypotheses, max_n=4)
    n_images = len(stripped_hypotheses)
    if n_images == 0:
        raise ValueError("Cannot evaluate an empty prediction set")
    return {
        "bleu_1": bleu[1],
        "bleu_2": bleu[2],
        "bleu_3": bleu[3],
        "bleu_4": bleu[4],
        "rouge_1": sum(
            rouge_n(h, r, 1) for h, r in zip(stripped_hypotheses, stripped_references)
        )
        / n_images,
        "rouge_2": sum(
            rouge_n(h, r, 2) for h, r in zip(stripped_hypotheses, stripped_references)
        )
        / n_images,
        "rouge_l": sum(
            rouge_l(h, r) for h, r in zip(stripped_hypotheses, stripped_references)
        )
        / n_images,
        "average_caption_length": sum(len(h) for h in stripped_hypotheses) / n_images,
        "unk_count": sum(unk_count(h) for h in stripped_hypotheses),
    }
