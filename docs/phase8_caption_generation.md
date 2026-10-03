# Phase 8 — Caption Generation

## 1. Objective

Generate a caption for any input image by joining the project's two
branches: the **visual branch** (Phases 2–5) finds visually similar
training images; the **caption branch** (Phases 6–7) supplies their
training captions and a trigram language model; a deterministic greedy
decoder turns the combination into one caption. Nothing new is learned in
this phase — no training, no neural components, no beam search.

## 2. Overall architecture

```
Image
 ↓  Phase 2 preprocessing (128×128, RGB/grayscale/HSV)
 ↓  Phase 3 classical features (HOG 8100 + Color 96 + LBP 10)
 ↓  Phase 4 scalers + fusion + PCA (transform only)
 ↓  1,804-D query vector
 ↓  Phase 5 cosine retrieval over the 6,000-image training gallery
 ↓  Top-K visually similar training images (default K = 5)
 ↓  their TRAINING captions (≤ 5·K candidate captions, e.g. 25)
 ↓  candidate words (+ <END>)
 ↓  Phase 7 trigram model + retrieval-evidence bonus
 ↓  greedy deterministic decoding
Generated caption
```

Crucial design principle: **the n-gram model never sees the image.** The
visual component contributes *which* training captions become candidates;
the language model decides *which word sequence* is most probable. The
bridge is lexical evidence extracted from the neighbours' captions.

## 3–6. Image side (reuse, not duplication)

- **Preprocessing (Phase 2)** — `preprocess_image`: 128×128 Lanczos
  resize, RGB / grayscale (Rec. 709) / HSV float arrays.
- **Classical features (Phase 3)** — `extract_features`: HOG (8,100),
  HSV colour histogram (96), uniform LBP (10). These are low-level
  descriptors: edges, colour distribution, texture. They do **not**
  recognise objects.
- **Fusion + PCA (Phase 4)** — the *already-fitted* per-family scalers and
  PCA transform the query to the 1,804-D space. Nothing is refit.
- **Retrieval (Phase 5)** — `ImageRetriever.retrieve`: cosine similarity
  against the 6,000-image training gallery, deterministic Top-K with
  self-exclusion.

## 7. Caption candidate extraction

`candidates.py` maps each retrieved `image_id` to its 5 training captions
(Phase 6 token sequences, `<START>`…`<END>` included) in deterministic
order (retrieval rank, then caption order). With K = 5 that is up to
**25 candidate captions**. A retrieved image without training captions
raises an error — the gallery is training-only, so absence would mean a
broken pipeline, never a silent skip. The candidate **word list** is the
unique tokens of those captions (`<START>`/`<PAD>` excluded, `<END>`
kept), ordered by caption-frequency descending with alphabetical
tie-breaks — a total order the decoder can rely on.

## 8. N-gram language model

The Phase 7 trigram (Laplace α = 1.0, V = 7,374, 132,845 fitted trigrams)
is loaded from `data/language_model/ngram_model.json` and used in
transform-only mode:

    score(w | context) = log P(w | last two tokens) + visual_weight · log evidence(w)

`evidence(w)` = fraction of candidate captions containing `w` — the
**retrieval-evidence bonus**. A word echoed by many retrieved neighbours
gets bonus → 0; a word from a single neighbour is penalised (fraction 1/25
≈ −3.2 in log terms); `<END>`, contained in every caption, gets bonus 0.
`visual_weight = 0` disables the bonus (pure language model); the default
is 1.0. This term is the entire "connection between retrieval and
language" — simple, explainable, deterministic.

## 9–10. Greedy decoding and candidate scoring

Starting from `tokens = [<START>]`, repeat:

1. filter candidates: drop the immediately previous word (no
   `dog dog`), drop words already used `max_word_repetitions` times
   (default 2), drop `<END>` until `min_length` words exist (default 3);
2. score every remaining candidate (formula above);
3. append the highest-scoring word (strict `>` — ties go to the
   deterministic candidate order);
4. stop if it was `<END>`, or when `max_length` (default 20) is reached.

If every candidate is filtered out, decoding stops early — deterministic,
documented. The final caption never contains `<START>`, `<PAD>` or `<END>`
(they are boundary/control tokens, removed from the output).

## 11. Repetition handling

Purely rule-based (no learned mechanism): immediate repetition is banned;
each word may appear at most `max_word_repetitions` times per caption;
`<START>`/`<PAD>` are globally banned. This prevents pathological outputs
like "dog dog dog …" deterministically.

## 12. Stopping conditions

Generation stops when (a) the decoder selects `<END>`, (b) `max_length`
words have been generated, or (c) the filtered candidate pool is empty.
The `<END>` token itself is removed from the displayed caption.

## 13. Data leakage prevention

- Candidates come **only** from training images: the Phase 5 gallery is
  training-only, the caption map is built from `train_captions.csv` only,
  and a retrieved image without training captions is an error.
  Verified by test (synthetic: non-retrieved images' words cannot appear;
  real: gallery IDs ⊆ training split IDs and every generated word is in
  the candidate pool of the retrieved images).
- The Phase 7 model is loaded, never refit; its fitting data was the same
  30,000 training captions.
- Validation/test captions never enter any Phase 8 code path.

## 14. Real behaviour (measured, honestly reported)

Query `1000268201_693b08cb0e.jpg` (girl in a pink dress on wooden porch
steps), K = 5 → 25 candidate captions, 117 candidate words → generated
caption: **"a man in a field"**. Query `1001773457_577c3a7d70.jpg` (two
dogs on asphalt) → **"a man in a blue and white"** (decoder chose `<END>`
after 6 words). Both deterministic across runs.

This is the expected behaviour of the architecture, and it must be stated
plainly: because Phase 5 neighbours are only *visually* similar (see
Phase 5 §6), their captions are about other scenes; the trigram then
produces a locally plausible but generic Flickr8k-style sentence — "a man
in a …" is the corpus's most probable opening. Caption *quality* therefore
depends heavily on retrieval quality and on the (weak, measured in Phase
7) smoothing; both are documented limitations, to be quantified in Phase 9
with BLEU/ROUGE/METEOR.

## 15. Limitations

1. HOG, colour histogram and LBP are **low-level visual descriptors** — no
   semantic object recognition.
2. Retrieval is **visual-similarity** based; neighbours need not depict
   the same objects or actions.
3. The n-gram model captures only **local word dependencies**; trigrams
   cannot model long-range language structure.
4. Classical n-grams **suffer from sparsity**; with Laplace smoothing the
   model is weak (validation PP measured in Phase 7), which encourages
   generic captions.
5. Caption quality is bounded by the retrieved captions — garbage
   neighbours, garbage candidates.
6. Greedy decoding is myopic; no lookahead (beam search deliberately
   excluded).
7. This is an intentionally **classical, explainable** system; it is not
   equivalent to modern neural captioning models and no such claim is
   made.

## 16. Usage

```bash
python -m src.caption_generation.run --image 1000268201_693b08cb0e.jpg
python -m src.caption_generation.run --image 1000268201_693b08cb0e.jpg --top-k 10 --max-length 15
```

## 17. Conclusion

Phase 8 completes the pipeline: an image in, a deterministic caption out,
through five explainable classical stages and one greedy decoder over a
fixed vocabulary. Every stage is separately tested (191 tests passing) and
every design decision is documented — the system is exactly as
interpretable as intended, with its limitations stated rather than hidden.
