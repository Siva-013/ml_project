# Phase 7 — N-gram Language Model

## 1. Objective

Build the caption-side language model: a classical statistical n-gram model
that estimates `P(next word | previous words)` from the Phase 6 training
captions. One generalized class (`NGramLanguageModel`) covers unigram,
bigram and trigram; the **trigram is the project's primary model**. No
caption generation happens here — Phase 8 consumes this model.

## 2. Input data (Phase 6 outputs, read-only)

```
data/captions/train_captions.csv        30,000 token sequences  → fitting
data/captions/validation_captions.csv    5,000 sequences        → perplexity only
data/captions/vocabulary.json            7,374 tokens           → fixed V, <UNK> mapping
```

The test split is not touched in Phase 7 — it stays reserved for Phase 9.

## 3. Training / validation separation (data-leakage prevention)

- **Fitting** (all n-gram counts, total token count, smoothing inputs)
  uses the 30,000 training captions only.
- **Validation** (5,000 captions) is only ever *scored* (perplexity) and
  used for smoothing exploration — never fitted.
- A canary test proves the point: a word appearing only in held-out
  captions produces zero new counts, maps through `<UNK>`, and never
  expands the vocabulary.

## 4. What is an n-gram?

An n-gram is a contiguous sequence of n tokens. A caption
`<START> a dog is running <END>` yields:

| Order | Windows |
| --- | --- |
| unigram | `<START>` · a · dog · is · running · `<END>` |
| bigram | (`<START>`,a) (a,dog) (dog,is) (is,running) (running,`<END>`) |
| trigram | (`<START>`,a,dog) (a,dog,is) (dog,is,running) (is,running,`<END>`) |

## 5–7. Unigram, bigram, trigram

- **Unigram** — probability of each word alone: `P(w)`.
- **Bigram** — one word of context: `P(w_i | w_(i-1))`.
- **Trigram** — two words of context (primary model): `P(w_i | w_(i-2), w_(i-1))`.
  Longer context = more specific conditioning, but sparser counts.

## 8. N-gram counting

`collections.Counter` over sliding windows of every training sequence
(`<PAD>` filtered before counting):

```
count_k(t₁..t_k) = number of positions where the k-token window occurs
```

**Measured on the real training data:**

| Order | Distinct n-grams |
| --- | ---: |
| unigrams | 7,372 |
| bigrams | 58,589 |
| trigrams | 132,845 |

(7,372 < 7,374 because `<PAD>` and `<UNK>` never occur in training text.)
Training tokens: 384,466 total (≈12.8 per caption; `<START>`/`<END>` each 30,000).

## 9. Maximum likelihood estimation

```
P(w_i | w_(i-2), w_(i-1)) = count(w_(i-2), w_(i-1), w_i) / count(w_(i-2), w_(i-1))
P(w_i | w_(i-1))          = count(w_(i-1), w_i)         / count(w_(i-1))
P(w_i)                    = count(w_i)                  / total_tokens
```

The denominator is the count of the context itself (`get_context_count`);
for the unigram case it is `total_tokens`.

## 10. The zero-frequency problem

Any unseen n-gram gets MLE probability 0 — and a product of sentence
probabilities containing a single 0 becomes 0, which would make most
validation captions *impossible*. Held-out data is guaranteed to contain
unseen n-grams (measured: the trigram table covers 132,845 of
7,374³ ≈ 4×10¹¹ possible trigrams), so smoothing is mandatory.

## 11. Laplace (add-alpha) smoothing

```
P(w | h₁, h₂) = (count(h₁, h₂, w) + α) / (count(h₁, h₂) + α·V)
P(w | h)      = (count(h, w) + α)      / (count(h) + α·V)
P(w)          = (count(w) + α)         / (total_tokens + α·V)
```

`α` is configurable (`--alpha`, default **1.0** = classic Laplace); `V` is
the fixed Phase 6 vocabulary size (7,374). Every event — seen or unseen —
therefore has strictly positive probability.

## 12. Sentence probability

```
P(<START> a dog runs <END>)
  = P(a|<START>) · P(dog|<START>,a) · P(runs|a,dog) · P(<END>|dog,runs)
```

At the very start the full trigram context does not exist yet, so the
first prediction uses the lower-order (bigram) counts — the shorter
context simply selects the corresponding order's counts. Contexts are
truncated to the last `n−1` tokens on the other end.

## 13. Log probability

Products of many small probabilities underflow quickly, so all sentence
scores are computed in log space (`sentence_log_probability` sums
`log P`); `sentence_probability` is `exp` of that sum. With Laplace
smoothing every P > 0, so logs are always finite (`alpha = 0` would allow
`-inf` and is guarded).

## 14. Perplexity

```
PP(W) = exp( -(1/N) · Σ log P(w_i | context_i) )
```

- `N` = number of **predicted** tokens = all tokens except each caption's
  leading `<START>` (so `<END>` counts, `<UNK>` counts as a normal token,
  `<PAD>` is filtered).
- Lower is better. Measured on the 5,000 validation captions (α = 1.0):

| Model | Validation perplexity |
| --- | ---: |
| Unigram | 239.08 |
| Bigram | 197.71 |
| **Trigram (primary)** | **659.28** |

**Why is the trigram worse?** This is the classic, textbook failure mode
of *add-1 smoothing on sparse higher-order models* (Jurafsky & Martin
highlight exactly this). Add-α reserves α·V of the denominator's mass for
unseen continuations; with V = 7,374 that is an enormous tax, and trigram
contexts are much sparser than bigram contexts (132,845 observed trigrams
over ~4×10¹¹ possibilities), so most validation trigram events receive the
near-uniform 1/(count+7,374). Measured α sweep (validation used for
selection only — test untouched):

| α | Trigram PP | Bigram PP (α) |
| ---: | ---: | ---: |
| 0.01 | **135.01** | — |
| 0.1 | 247.85 | **84.29** |
| 0.5 | 481.05 | — |
| 1.0 | 659.28 | 197.71 |

Conclusion for the report/viva: Laplace smoothing is pedagogically clear
but statistically crude; the per-order comparison quantifies how badly
add-smoothing punishes higher orders. Better schemes (interpolation with
lower orders, Kneser–Ney) are the known remedy — noted as future work, not
implemented, to keep the classical baseline exactly as specified. The
saved default model remains the trigram at α = 1.0.

## 15. Special tokens

| Token | Role in Phase 7 |
| --- | --- |
| `<PAD>` | filtered before counting and scoring — contributes to nothing (V still counts it, per the fixed Phase 6 vocabulary) |
| `<UNK>` | valid token; unseen words map to it at scoring time; never seen in training (count 0) |
| `<START>` | conditions the first word; never itself predicted |
| `<END>` | predicted like a word — this is what will let Phase 8 stop generating |

## 16. Unknown-token handling

The model holds the fixed Phase 6 vocabulary (token set). At probability
time, any token outside it — word or context word — is mapped to `<UNK>`
before lookup. The vocabulary and counts are never expanded (tested).

## 17. Data leakage prevention

- `NGramLanguageModel.fit` only ever receives the training sequences
  (structural guarantee — `train.py` passes `train_captions.csv` only).
- Validation captions flow exclusively into `evaluate_perplexity`.
- Canary tests: a validation-only word creates no counts, no vocabulary
  entries; refitting resets rather than accumulates.
- Phase 6 artifacts are read-only inputs; Phase 7 writes only
  `data/language_model/`.

## 18. Model persistence

```
data/language_model/
├── ngram_model.json      primary trigram: counts per order (keys are
│                         space-joined tokens — unambiguous because the
│                         Phase 6 tokenizer never produces spaces),
│                         α, total tokens; ~5 MB, human-inspectable
└── model_metadata.json   model type, n, smoothing, α, V, training count,
                          training_only: true, n-gram counts, validation
                          perplexity, per-order comparison, provenance
```

Deterministic: same inputs → byte-identical files (counts are sorted
before serialization). `load_model` reconstructs the model exactly.

## 19. Limitations

- **Add-α smoothing is weak** (measured above); interpolated or
  Kneser–Ney smoothing would improve perplexity substantially — future
  work within the classical toolbox.
- **Exact-sparse representation**: naive storage of seen n-grams only; no
  backoff weights, so unseen contexts get near-uniform estimates.
- **Markov assumption**: history beyond n−1 tokens is invisible; long-range
  structure ("a man … he") is not modeled.
- **No semantics**: the model knows nothing about images or meaning —
  identical treatment of related and unrelated words.
- **Vocabulary frozen at Phase 6**: genuinely new test-time words are
  `<UNK>` by design.

## 20. Conclusion

Phase 7 delivers a deterministic, fully inspectable trigram language model
(with unigram/bigram for comparison) fitted strictly on the 30,000 training
captions: 7,372 / 58,589 / 132,845 distinct n-grams, Laplace smoothing with
configurable α, log-space sentence scoring, and perplexity evaluation on
the validation split only. The measured per-order comparison honestly
documents the weakness of add-α smoothing on higher orders — valuable
analysis for the viva and a clear pointer for refinement. Input contract
for Phase 8 (caption generation): established.
