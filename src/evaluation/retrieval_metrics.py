"""Phase 9 — retrieval evaluation proxies.

Flickr8k provides no ground-truth "this image looks like that image" labels
between *distinct* images, so retrieval relevance cannot be measured as an
exact task. What Flickr8k does provide, for every image, is five human
captions — two different images showing similar content tend to have
captions sharing vocabulary ("dog", "grass", "running"). This module
therefore reports an explicit, weak **caption-overlap proxy**:

* Jaccard overlap between the query image's reference-caption token set
  and a retrieved neighbour's caption token set (function words included —
  the number is a proxy, not a semantic judgement);
* containment = |query tokens ∩ neighbour tokens| / |query tokens|.

These numbers must be reported as proxies. They measure whether retrieval
surfaces images whose *descriptions* resemble the query's descriptions —
nothing more. Exact self-match cannot occur for validation/test queries
because the gallery is training-only (asserted, reported as 0 by design).
"""

from __future__ import annotations

from typing import Iterable, Sequence


def caption_token_set(captions: Iterable[Sequence[str]]) -> set[str]:
    """Union of the token sets of several captions."""
    tokens: set[str] = set()
    for caption in captions:
        tokens.update(caption)
    return tokens


def jaccard_overlap(query_tokens: set[str], neighbour_tokens: set[str]) -> float:
    """|A ∩ B| / |A ∪ B| (0 if both empty)."""
    if not query_tokens and not neighbour_tokens:
        return 0.0
    union = query_tokens | neighbour_tokens
    return len(query_tokens & neighbour_tokens) / len(union) if union else 0.0


def containment(query_tokens: set[str], neighbour_tokens: set[str]) -> float:
    """|A ∩ B| / |A| — how much of the query's vocabulary the neighbour shares."""
    if not query_tokens:
        return 0.0
    return len(query_tokens & neighbour_tokens) / len(query_tokens)


def retrieval_relevance(
    query_reference_captions: Sequence[Sequence[str]],
    retrieved_captions: Sequence[Sequence[str]],
) -> dict[str, float]:
    """Caption-overlap proxies for one query against its retrieved captions."""
    query_tokens = caption_token_set(query_reference_captions)
    neighbour_tokens = caption_token_set(retrieved_captions)
    return {
        "jaccard_overlap": jaccard_overlap(query_tokens, neighbour_tokens),
        "containment": containment(query_tokens, neighbour_tokens),
    }
