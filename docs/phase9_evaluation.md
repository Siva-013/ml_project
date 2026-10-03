# Phase 9 — Experimental Evaluation and Algorithm Comparison

## 1. Objective and discipline

Measure the complete Phase 1–8 pipeline honestly, including poor results.
Split discipline (documented, enforced by workflow):

- **Training** (6,000 images / 30,000 captions) — only for fitting the
  existing models. Nothing new is fitted in Phase 9 except unigram/bigram
  comparison models on the *same* training data.
- **Validation** (1,000 images / 5,000 captions) — all experiments and all
  configuration choices (model order, Top-K).
- **Test** (1,000 images / 5,000 captions) — evaluated **once**, with the
  configuration fixed from validation; results were not used to change
  anything afterwards.

No random sampling is used anywhere; every selection is deterministic
(ranking-based), so no seed needs to be recorded.

## 2. Evaluation setup

| Item | Value |
| --- | --- |
| Visual representation | HOG 8,100 + HSV hist 96 + LBP 10 → fused 8,206 → PCA 1,804 (train-fitted) |
| Retrieval | cosine similarity, training-only gallery (6,000) |
| Top-K values evaluated | 1, 3, 5, 10 |
| Language models | unigram, bigram, trigram (Laplace α = 1.0) |
| Decoding | Phase 8 greedy (max 20 words, min 3), reused unchanged |
| Metric tokenization | Phase 6 normalization (lowercase, punctuation → spaces, split) |
| Special tokens | `<START>`/`<END>`/`<PAD>` stripped before scoring; `<UNK>` counted and reported (Phase 8 candidates never produce it — measured: 0) |
| BLEU convention | corpus-level, cumulative BLEU-k = BP × geometric mean(p₁…p_k); multi-reference clipping against max reference count; closest-reference-length brevity penalty |
| ROUGE convention | per-image max-F1 over the 5 references (ROUGE-1, ROUGE-2, ROUGE-L via LCS); corpus score = mean over images |

## 3. Experiment 1 — visual retrieval (proxy metrics)

Flickr8k has no inter-image ground-truth similarity labels, so retrieval
is evaluated with an explicit weak proxy: **caption-vocabulary overlap**
between the query's five reference captions and the retrieved neighbours'
captions (Jaccard over the K neighbours' union; containment against the
query's vocabulary). This measures whether retrieval surfaces images whose
*descriptions* resemble the query's descriptions — nothing more.

Measured on 1,000 validation queries:

| K | mean similarity | Jaccard proxy | containment proxy | exact self-matches |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.2573 | 0.1265 | 0.2273 | 0 |
| 3 | 0.2437 | 0.1217 | 0.3716 | 0 |
| 5 | 0.2362 | 0.1074 | 0.4482 | 0 |
| 10 | 0.2248 | 0.0854 | 0.5567 | 0 |

Reading: containment grows with K (more neighbours cover more of the
query's vocabulary) while mean similarity and Jaccard fall (lower-ranked
neighbours are farther away). Exact self-matches are 0 **by construction**
— validation/test images are not in the training gallery.

## 4. Experiments 2–4 — caption generation quality

Primary configuration: trigram, K = 5.

| Metric | Validation | Test |
| --- | ---: | ---: |
| BLEU-1 | 0.5042 | 0.4884 |
| BLEU-2 | 0.2889 | 0.2857 |
| BLEU-3 | 0.1530 | 0.1566 |
| BLEU-4 | 0.0839 | 0.0900 |
| ROUGE-1 | 0.3726 | 0.3680 |
| ROUGE-2 | 0.1224 | 0.1238 |
| ROUGE-L | 0.3441 | 0.3432 |
| Avg caption length | 8.33 words | 8.23 words |
| `<UNK>` count | 0 | 0 |

Test tracks validation closely — the configuration selected on validation
did not overfit. These BLEU-4 values (~0.08–0.09) are low in absolute
terms and are reported as measured; they are consistent with a purely
classical pipeline whose retrieval is visual-statistical rather than
semantic (see Phase 5 §6 and Phase 8 §14).

## 5. Experiment 5 — language-model comparison (validation)

| Model | Val. perplexity | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-1 | ROUGE-2 | ROUGE-L | Avg len |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unigram | 239.08 | 0.2218 | 0.0911 | 0.0000 | 0.0000 | 0.3730 | 0.0505 | 0.3584 | 3.00 |
| Bigram | 197.71 | 0.4941 | 0.2810 | 0.1370 | 0.0701 | 0.3610 | 0.1139 | 0.3387 | 8.83 |
| Trigram | 659.28 | 0.4837 | 0.2747 | 0.1459 | 0.0812 | 0.3635 | 0.1159 | 0.3374 | 8.07 |

(`<UNK>` count 0 for every model. Generation "failure rate" — captions
that exit early via pool exhaustion — shows up in the unigram's length:
it produces exactly 3-word captions every time, its minimum length.)

Three findings, reported without calling any model "best":

1. **Perplexity and generation quality rank differently.** The bigram has
   the *lowest* perplexity (197.7) yet the trigram scores the *highest*
   BLEU-3/BLEU-4 (0.1459/0.0812) while the bigram edges BLEU-1/BLEU-2
   (0.4941/0.2810). Perplexity measures how well a model predicts held-out
   *reference text*; the generation metrics measure how well the *decoded
   output* — constrained to retrieval candidates and repetition rules —
   matches references. Different properties, different rankings; reported
   side by side, never merged into one score.
2. **The unigram cannot produce structured captions**: BLEU-3/4 = 0 by
   construction (no word-order information; 3-word minimum-length outputs
   contain no trigrams or 4-grams). Yet its ROUGE-1 (0.3730) *edges the
   trigram's* (0.3635) — unigram overlap rewards the frequent-word
   captions it emits. Another demonstration that single metrics disagree.
3. The trigram trades short-sentence precision (BLEU-1) for higher-order
   structure (BLEU-3/4) — the expected effect of conditioning on two
   context tokens.

## 6. Experiment 6 — Top-K comparison (validation, trigram)

| K | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-1 | ROUGE-2 | ROUGE-L | Avg len |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.4312 | 0.2328 | 0.1204 | 0.0638 | 0.3387 | 0.0935 | 0.3189 | 6.87 |
| 3 | 0.4819 | 0.2743 | 0.1407 | 0.0765 | 0.3651 | 0.1158 | 0.3407 | 7.71 |
| 5 | 0.5042 | 0.2889 | 0.1530 | 0.0839 | 0.3726 | 0.1224 | 0.3441 | 8.33 |
| 10 | 0.5087 | 0.2955 | 0.1632 | 0.0950 | 0.3777 | 0.1319 | 0.3460 | 9.38 |

Every metric increases monotonically with K, with strongly diminishing
returns: K = 1 → 5 gains ≈ +0.07 BLEU-1, K = 5 → 10 gains ≈ +0.005
(≈ +1 word of average length per few extra candidates). Larger pools
help, at linear retrieval/decode cost — stated as measurement, not as a
verdict.

## 7. Experiment 7 — qualitative analysis (deterministic selection)

30 validation images (10 highest / 10 median / 10 lowest per-image
ROUGE-L) with references, retrieved neighbours, similarities and heuristic
failure categories (categories reported only when triggered):

- observed: `generic` (19/30 — overwhelmingly the "a man in a …" template
  family), `low_caption_overlap` (7/30 — neighbours' captions share little
  vocabulary with the query's references), `none` (8/30);
- **good** example: generated *"a dog is running in the water"* vs
  reference *"a brown dog runs through the water as a wave comes"*;
- **poor** example: generated *"a man in a white dog is in the water"* vs
  reference *"two men in magenta costumes performing"* (generic +
  low-overlap retrieval);
- not observed, hence not reported: `<UNK>` generation, pathological
  repetition (both prevented by Phase 8 construction).

## 8. Experiment 8 — external qualitative evaluation

No external images were supplied for this run, so
`data/evaluation/external_evaluation.csv` is written empty and the run
summary records "images evaluated: 0". The CLI supports evaluating
arbitrary user images at any time (`python -m src.caption_generation.run
--image <path>`), without metrics (no references exist). Any such
evaluation is labelled **external qualitative evaluation** and is not
statistically comparable to the test-set results.

## 9. Experiment 9 — data-leakage checks (automated, all PASS)

| Check | Result |
| --- | --- |
| Vocabulary built from training captions only | PASS |
| Candidate captions from training images only | PASS |
| StandardScaler fitted on training features only (`n_samples_seen_ = 6000`) | PASS |
| PCA fitted on training features only (metadata `train_samples = 6000`) | PASS |
| Train/validation/test separation (6,000/1,000/1,000, unassigned 91 excluded) | PASS |
| Exact self-retrieval of validation/test queries | 0 (gallery is training-only, by construction) |
| Test references used only in the final test evaluation | by workflow (validation ran first; test ran once) |

## 10. Experiment 10 — error analysis

The 30 qualitative examples separate into:

- **Good (8 of 30 clean):** retrieval surfaced dogs/scenes matching the
  query, and the trigram assembled a locally-correct sentence
  ("a dog is running in the water").
- **Language-model failures:** the dominant pattern (19/30) is the
  **generic template** "a man in a …" — the corpus's most probable
  openings win whenever retrieval evidence is weak. The Laplace-smoothed
  trigram's weakness (Phase 7) directly causes this.
- **Visual retrieval failures:** 7/30 examples have neighbours whose
  captions share almost no vocabulary with the query's references —
  color/texture-driven retrieval pulls in unrelated scenes, and no
  language model can recover from wrong candidates.
- **End-to-end:** the poor examples are predominantly the compound case —
  weak retrieval *and* weak LM, compounding. The good examples show the
  chain works when both stages happen to align.

## 11. Reproducibility

Everything is deterministic: no random sampling (qualitative blocks are
rank-based), fixed evaluation order, and the metrics/pipelines are covered
by determinism tests. Outputs in `data/evaluation/`:

```
validation_predictions.csv    validation_metrics.{json,csv}
test_predictions.csv          test_metrics.{json,csv}
retrieval_results.csv         language_model_comparison.csv
top_k_comparison.csv          qualitative_examples.csv
external_evaluation.csv       evaluation_summary.json
```

## 12. Suitability conclusion (factual)

Measured findings: BLEU-4 ≈ 0.08–0.09 / ROUGE-L ≈ 0.34 on test; retrieval
proxies show weak-to-moderate caption overlap; failure analysis attributes
most poor outputs to generic LM behaviour plus visually-driven retrieval.
The classical pipeline is **functional, fully explainable, and
leakage-free**, and it produces plausible Flickr8k-style captions; its
semantic grounding is weak by construction of its handcrafted features.
Whether that is "suitable" depends on the objective — for an explainable
classical baseline it meets the brief; as a caption *quality* system it
sits far below modern neural systems, as the measurements make clear.

## 13. Limitations of the evaluation itself

- The retrieval proxy (caption overlap) is weak supervision, not ground
  truth.
- BLEU/ROUGE measure n-gram overlap with references, not factual
  correctness.
- Single test-set run; no confidence intervals (deterministic corpus
  metrics; bootstrap could be added).
- External-image evaluation was empty (no user images supplied).
