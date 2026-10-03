# Phase 5 — Image Retrieval

## 1. What is image retrieval?

Retrieval answers: *given a query image, which gallery images look most like
it?* Formally:

```
query image
      ↓  feature representation (here: the Phase 4 PCA space)
similarity comparison against every gallery vector
      ↓
ranked list of gallery images (Top-K)
```

In this project retrieval is the bridge between vision and language: its
Top-K results are what a later phase will attach human captions to. Nothing
here is learned at query time — retrieval is pure comparison.

## 2. Why cosine similarity?

```
cos(q, x) = (q · x) / (||q|| ||x||) ∈ [-1, 1]
```

Cosine measures the **angle** between vectors, ignoring their magnitudes.
That fits this pipeline: Phase 4 already standardized every feature family,
so magnitudes carry little meaning (a brighter image must not rank higher
just because its raw values are larger); direction in the PCA space is the
informative part. It is also cheap — with both sides pre-normalised it
reduces to one matrix–vector product, O(N·D), vectorised in NumPy (no
FAISS or external retrieval framework needed for N = 6,000).

Implementation: the gallery is L2-normalised once at construction; each
query is L2-normalised after the PCA projection; similarity =
`gallery_normalized @ query_normalized`.

## 3. Why the gallery is training images only

The gallery is the **6,000-image training split** — never validation, test
or unassigned images. Rationale (data-leakage):

- During Phase 9 evaluation, test images are the queries. If test images
  were in the gallery, a query could retrieve *itself* (similarity ≈ 1)
  or a near-duplicate, and the captions copied from such neighbours would
  inflate every metric — a classic leakage failure.
- The same rule will govern Phase 8: generated captions must be built from
  *training* human captions only.

Verified by test: the gallery IDs equal exactly
`image_ids[splits == "train"]`, in the original Phase 3 order.

## 4. Connection to Phase 4

```
HOG (8100) + Color (96) + LBP (10)
      ↓  Phase 4 scalers (transform only)      z = (x - μ)/σ, train μ/σ
fusion → 8,206 dimensions
      ↓  Phase 4 PCA (transform only)
1,804-dimensional query/gallery vectors
      ↓  cosine similarity + ranking
Top-K results
```

The retriever **loads** `pca.joblib` and the three `scalers/*.joblib` saved
by Phase 4 and never fits anything: `extract_query_vector` reuses the
Phase 2 preprocessing, Phase 3 extractors and the Phase 4
`transform_and_fuse`, then applies the saved PCA. The query representation
is exactly `(1, 1804)`.

**Self-retrieval behaviour:** if the query is itself a gallery member (a
training image), it is excluded from its own results by default
(`exclude_self=True`) — otherwise rank 1 would always be the identical
image at similarity ≈ 1. For validation/test queries this is a no-op (they
are not in the gallery). Pass `--include-self` / `exclude_self=False` to
disable; the self-match then appears at rank 1 with similarity > 0.999
(verified by test).

## 5. Retrieval complexity

- **Query transformation:** Phase 2 preprocessing + Phase 3 features
  (~15 ms) + scaler/PCA transforms (one 1×8,206 matrix-vector product
  each) — per query.
- **Similarity:** O(N × D) = 6,000 × 1,804 ≈ 10.8M multiply–adds —
  sub-millisecond in NumPy. Gallery normalisation is done once at
  construction, not per query.
- **Ranking:** `lexsort` over N similarities — O(N log N), with image ID as
  the deterministic tie-breaker.

## 6. Actual retrieval behaviour (qualitative, real queries)

Structural results (all verified by tests): gallery = 6,000 × 1,804, IDs
aligned, results deterministic across repeated queries, similarities within
[-1, 1], descending order, invalid K rejected, self-exclusion works.

Two real queries, recorded honestly:

- **Query** `1001773457_577c3a7d70.jpg` (two dogs on an asphalt road) →
  Top-5 are rock-climbing, dirt-bike-track, rocky-slope, asphalt-toddler
  and snowy-slope photos (similarities 0.199–0.230). They share the
  grey/brown asphalt-and-earth *texture and tones* of the query, but are
  not semantically "dogs".
- **Query** `1000268201_693b08cb0e.jpg` (girl in a pink dress on wooden
  porch steps) → Top-5 are outdoor people/street scenes (similarities
  0.150–0.165), sharing rough composition and tones but not the semantics.

**Honest reading:** the retriever ranks by low-level visual statistics
(colour distribution, texture, edge statistics), as documented since
Phase 3. Cosine values in the 1,804-D standardized space are small even
for visually related images (vectors are near-orthogonal); the *ranking* is
the meaningful signal, not the absolute similarity value. This is the
expected behaviour of the classical pipeline and the main risk inherited by
caption generation in Phase 8 — to be measured, not assumed, in Phase 9.
See `outputs/visualizations/retrieval_*.png` for the rendered examples.

## 7. Usage

```bash
python -m src.retrieval.run_retrieval --image 1000268201_693b08cb0e.jpg --top-k 5
python -m src.retrieval.run_retrieval --image D:/path/to/photo.jpg --top-k 10
python -m src.retrieval.visualize --image 1000268201_693b08cb0e.jpg --top-k 5
```

## 8. Limitations

- Classical features retrieve by **colour, texture, edges and shape
  statistics** — not by objects, actions or scene meaning. Two images with
  similar palettes and textures can depict entirely different things (see
  §6), and two images of the same object type with different palettes may
  rank far apart.
- Cosine similarity in the PCA space treats all retained variance
  directions equally; no relevance weighting exists at this stage.
- The linear PCA projection cannot model non-linear visual similarity.
- Retrieval quality is bounded by the Phase 3 features; improving semantic
  behaviour would require changing the representation (out of scope for
  the classical baseline).

## 9. Conclusion

Phase 5 delivers a deterministic, vectorised, training-only classical
retriever: 6,000-image gallery in the 1,804-D Phase 4 space, cosine
similarity with deterministic tie-breaking, reusable Phase 4
transformations, self-exclusion for gallery-member queries, 111 tests
passing end to end. Input contract for Phase 6–8 (caption processing,
n-gram model, generation): established.
