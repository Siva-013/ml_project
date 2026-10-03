# Phase 4 — Feature Fusion and PCA

## 1. Why feature fusion?

Phase 3 produced three complementary, independent descriptions of every
image:

| Family | Captures | Blind to |
| --- | --- | --- |
| HOG (8,100) | edges, gradients, contours, local shape/structure | colour, global colour distribution |
| Colour histogram (96) | global colour distribution | where colours are, shape, texture |
| LBP (10) | local texture statistics | shape, colour, spatial layout |

No single family sees the whole picture: a red car and a red sofa are close
in colour space, a zebra and a white wall share brightness, and HOG is
colour-blind. **Fusion concatenates them into one vector so the
representation supports all three notions of visual similarity
simultaneously** — which later retrieval needs, because "visually similar"
images may match in shape (two dogs), colour (beach scenes) or texture
(foliage).

## 2. Why standardization?

The three families have wildly different numerical scales and
dimensionalities (8,100 vs 96 vs 10 features; HOG values are L2-Hys
normalised gradient histograms, colour/LBP values are small frequencies).
Raw concatenation would let HOG dominate the geometry by sheer magnitude
and dimension count.

Each family therefore gets its **own** `StandardScaler`, computed from
training statistics only:

```
z = (x - μ) / σ        μ = training-set mean,  σ = training-set std
```

After standardization every family contributes comparable z-scores, and the
PCA (which is variance-based) is not biased towards the widest-scaled
family. Verified on the training data: per-feature mean ≈ 0, std ≈ 1
(asserted in tests with float32 tolerance).

## 3. Feature fusion

```
standardized HOG (8100) + standardized Colour (96) + standardized LBP (10)
= 8206 dimensions
```

Plain `np.concatenate(..., axis=1)` in the fixed family order
`[hog, color, lbp]`. Fusing happens **after** standardization and **before**
PCA. Verified fused shapes: train (6000, 8206), validation (1000, 8206),
test (1000, 8206), unassigned (91, 8206).

## 4. Why PCA?

The fused 8,206-D vector is large and highly redundant (HOG's overlapping
blocks alone duplicate information). PCA compresses it while keeping the
information that actually varies across the dataset:

```
X (n × 8206)
   ↓  centre (scikit-learn does this internally)
covariance structure of the training data
   ↓  eigendecomposition (exact SVD)
principal directions = eigenvectors of the covariance, ordered by eigenvalue
   ↓  projection onto the top-k directions
compact representation (n × k)
```

Benefits in this project: **redundancy reduction**, **compact storage**
(8,206 → 1,804 float32 ≈ 4.5× smaller), **fast distance computations** for
the Phase 5 retrieval index, and **variance preservation** (the retained
directions are, by construction, the ones along which training images
differ most).

## 5. Data-leakage prevention (enforced by construction)

| Object | Fitted on | Everything else |
| --- | --- | --- |
| HOG scaler | 6,000 training rows only | transform only |
| Colour scaler | 6,000 training rows only | transform only |
| LBP scaler | 6,000 training rows only | transform only |
| PCA | 6,000 training fused rows only | transform only |

- The `fit_*` functions *only accept training arrays* — validation, test
  and unassigned rows physically cannot reach a fitting call.
- Proof from the artifacts: every saved scaler records
  `n_samples_seen_ = 6000` (asserted by tests on the real joblib files).
- Validation/test/unassigned rows are only ever transformed.
- The 91 unassigned images never influence any fit and are stored
  separately (`unassigned_features.npy` / `unassigned_image_ids.npy`).
- Behavioural canary in the unit tests: with non-training rows shifted far
  away in feature space, the fitted means still match the training means.

## 6. Actual PCA result (measured, not assumed)

```
Original dimensions : 8,206
PCA components      : 1,804          (pca.n_components_)
Explained variance  : 0.950047       (pca.explained_variance_ratio_.sum())
PCA fitted on       : train only (6,000 rows)
Solver              : full (exact, deterministic SVD; no randomness)
```

The component count was selected automatically from the 95 % variance
target applied to the real training data — never hard-coded. 1,804 of
8,206 dimensions (a 4.55× reduction, 78 % of dimensions removed) retain
95.00 % of the training variance. Determinism: the same inputs reproduce
the same transformation (asserted numerically); the first two components
carry 5.2 % and 2.0 % of the variance respectively (see
`outputs/visualizations/pca_scatter_pc1_pc2.png`, where train/validation/test
distributions overlap — expected, since the official split is not
visual-content-aware).

**Why full PCA and not IncrementalPCA:** the training fused matrix is
6,000 × 8,206 float32 ≈ 197 MB, which fits comfortably in memory; the exact
solver ran in ~90 s end-to-end. IncrementalPCA (approximate, batch-wise)
would add complexity without need.

## 7. Storage

```
data/features/fusion/
├── train_features.npy          (6000, 1804) float32
├── validation_features.npy     (1000, 1804) float32
├── test_features.npy           (1000, 1804) float32
├── unassigned_features.npy     (  91, 1804) float32
├── train_image_ids.npy / validation_image_ids.npy /
│   test_image_ids.npy / unassigned_image_ids.npy     (aligned row-wise)
├── pca.joblib                  fitted PCA — reused at inference, never refit
├── scalers/hog_scaler.joblib, color_scaler.joblib, lbp_scaler.joblib
└── fusion_metadata.json        all actual values, counts, provenance
```

Phase 3 files under `data/features/` are read-only inputs to this phase and
were not modified (verified by tests and file timestamps). Row `i` of every
`*_features.npy` corresponds to row `i` of the matching `*_image_ids.npy`,
in the original Phase 3 order (no sorting, no shuffling).

## 8. Limitations

- **PCA is unsupervised.** It knows nothing about objects, actions,
  relationships, captions, language or semantic meaning — it only finds the
  directions of maximum *numerical* variance in the handcrafted visual
  feature space. A direction that separates many images may still be
  visually meaningless (e.g. overall brightness).
- Variance ≠ relevance for captioning: the 5 % variance discarded could in
  principle contain discriminative detail; conversely, retained variance
  does not guarantee retrieval usefulness. Phase 5/9 will measure this
  empirically.
- Standardization makes every feature equally important per-dimension;
  within-family importance weighting is out of scope for the classical
  baseline.
- The projection is linear; non-linear structure in the visual space is not
  captured (no kernels — deliberately kept classical and cheap).

## 9. Conclusion

Phase 4 fuses the three Phase 3 families into an 8,206-dimensional
standardized representation and compresses it, using transformations fitted
exclusively on the 6,000 training images, to **1,804 dimensions retaining
95.00 % of training variance**. The fitted scalers and PCA are persisted
for reuse at inference, all matrices are aligned with image IDs, the 91
unassigned images are transformed but excluded from fitting, and the full
test suite passes (93 tests). Input contract for Phase 5 (KNN retrieval +
K-Means analysis on the training rows): established.
