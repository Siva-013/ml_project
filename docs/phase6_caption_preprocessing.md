# Phase 6 — Caption Preprocessing

## 1. Why caption preprocessing is necessary

Raw Flickr8k captions vary in capitalization ("A child…", "two women…"),
carry punctuation (mostly pre-separated: "… an entry way ."), and contain
whitespace variations. An n-gram model counts *token occurrences* — if
"Dog" and "dog" were different tokens, or "way ." were one token, the
counts (and therefore every conditional probability in Phase 7) would be
fragmented and wrong. Preprocessing turns every caption into a stable,
canonical token sequence so identical wording always produces identical
token identifiers.

Like everything else in this project, the pipeline is **deterministic and
classical**: string operations only — no learned components, no pretrained
tokenizers, no neural networks.

## 2. Normalization (rules actually implemented)

Applied in order, to every caption:

1. **Missing-value check** — `None`/NaN captions raise a `ValueError`
   (a missing caption is a data-integrity problem, never silently an empty
   caption; Phase 1 data contains none).
2. **Lowercase** — `"A Dog Runs"` → `"a dog runs"`.
3. **Punctuation → spaces** — every ASCII punctuation character
   (`string.punctuation`, documented set) is replaced by a space.
4. **Whitespace collapse** — runs of whitespace become one space; edges are
   stripped.
5. **No stemming / no lemmatization** — natural wording is preserved for
   later generation, per the academic baseline.

Punctuation-policy consequences (deterministic, pinned by tests):
Flickr8k's pre-separated sentence periods vanish cleanly
(`"an entry way ."` → `"an entry way"`); contractions split
(`"dog's"` → `"dog s"`) and hyphenated compounds split
(`"ten-year-old"` → `"ten year old"`) rather than gluing words together —
the standard classical trade-off.

## 3. Tokenization

After normalization, tokens are obtained with plain whitespace splitting
(`str.split()`): `"a dog runs"` → `["a", "dog", "runs"]`. A deterministic
word-level tokenizer is the right match for a classical n-gram model:
n-gram probabilities are defined over *word* sequences, and any
sub-word/pretrained tokenizer would break the countable-word semantics and
add a non-classical dependency.

## 4. Boundary tokens

Every processed caption is `<START> token₁ … tokenₙ <END>`:

- **`<START>`** gives the Phase 7 model a conditioning context for the
  first word — `P(w₁ | <START>)` — so generation can begin from nothing.
- **`<END>`** lets the model learn where captions stop, so Phase 8
  generation knows when to emit a stopping point instead of trailing off.

The exact strings `<START>`/`<END>` are used (plus `<PAD>`/`<UNK>` in the
vocabulary; padding is deliberately **not** applied — nothing in Phase 6
needs equal-length sequences).

## 5. Vocabulary

```
training captions (30,000)
      ↓  normalize → tokenize → boundaries
token frequency counts
      ↓  specials first (<PAD>0, <UNK>1, <START>2, <END>3),
      ↓  then frequency descending, ties alphabetical — deterministic
vocabulary (7,374 tokens)
```

`min_frequency = 1` (the academic baseline: keep the whole training
vocabulary). The threshold exists as configuration (`--min-frequency`) and
its effect is tested: tokens below it are excluded and would encode to
`<UNK>`. Most frequent training tokens: *a* (46,784), *in* (14,094),
*the* (13,509), *on* (8,007), *is* (6,907), *and* (6,678), *dog* (6,160).
`<START>`/`<END>` each occur exactly 30,000 times — once per training
caption.

## 6. Unknown words

Validation/test captions contain words the model has never seen. Any token
outside the training vocabulary encodes to the `<UNK>` ID — e.g. a
validation-only "zebra" becomes `<UNK>` (2,379 `train.csv` occurrences of
words that were simply absent from training would otherwise be impossible
to model). Validation/test-only tokens **never** enter the vocabulary;
this is verified directly by a test that checks every vocabulary token
occurs in the training captions.

## 7. Data leakage prevention

```
vocabulary ← TRAINING captions only (30,000)
validation (5,000) / test (5,000) captions → encoded with that fixed vocabulary
91 unassigned images → excluded from every Phase 6 output
```

The pipeline slices the training rows *before* calling `Vocabulary.build`
(the method only ever receives training sequences). Verified by tests:
metadata counts (8,000 images / 40,000 captions processed), split
integrity against the Phase 1 CSVs, absence of unassigned images, and the
vocabulary ⊆ training-tokens check. The `train_samples_seen`-style proof
here is structural: `Vocabulary.build` cannot see other splits because the
pipeline passes only `split == "train"` rows.

## 8. Integer encoding

Each token maps to a stable integer ID (specials 0–3, then
frequency-ranked words); every caption becomes an ID sequence aligned with
its `image_id`/`caption_id`, e.g. the first training caption encodes to
`[2, 4, 44, 5, …, 641, 3]`. Integer IDs are what Phase 7 needs: n-gram
contexts and outcomes become tuples of integers that index cleanly into
count tables, and lookups are O(1) dict operations.

## 9. Outputs

```
data/captions/
├── train_captions.csv        (30,000 rows)
├── validation_captions.csv   ( 5,000 rows)
├── test_captions.csv         ( 5,000 rows)
│       columns: image_id, caption_id, split, raw_caption,
│                clean_caption, tokens (JSON array string incl. <START>/<END>)
├── vocabulary.json           specials, token_to_id, id_to_token
│                             (JSON keys are strings), frequencies, size
├── encoded_captions.json     per split: {image_id, caption_id, token_ids}
└── caption_metadata.json     counts, rules, policies, provenance
```

Phase 1 files (`data/processed/*.csv`) are read, never modified. Running
the pipeline twice produces byte-identical outputs (tested).

## 10. Limitations

- **Word-level tokenization**: rare morphological variants ("run",
  "runs", "running") are distinct tokens, fragmenting counts; no
  stemming/lemmatization by design.
- **Unknown-word problem**: out-of-vocabulary validation/test words all
  collapse into `<UNK>` — information is lost; with `min_frequency = 1`
  this affects only genuinely unseen words.
- **No semantics**: preprocessing understands syntax at the character level
  only; "dog" and "cat" are equally unrelated tokens.
- **Training-data dependence**: the vocabulary is exactly as broad as the
  30,000 training captions (7,374 tokens); genuinely new words at test time
  cannot be represented.

## 11. Conclusion

Phase 6 converts the validated captions into deterministic token sequences
with boundary markers, a 7,374-token training-only vocabulary, and integer
encodings — a clean, leakage-free input contract for Phase 7 (n-gram
language model). The caption branch is now the structural twin of the
visual branch; they connect only in Phase 8.
