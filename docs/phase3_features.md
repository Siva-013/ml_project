# Phase 3 — Classical Visual Feature Extraction

## 1. Why handcrafted features?

This project intentionally avoids pretrained CNNs / Transformers. Three
reasons:

1. **Academic scope** — the project studies what classical CV + classical
   ML + statistical NLP can achieve without any deep learning, so every
   step must be explainable end-to-end (viva requirement).
2. **Explainability** — a HOG bin, an HSV histogram bin and an LBP code can
   be shown and interpreted directly; a CNN activation cannot.
3. **No external dependencies** — no pretrained weights, no downloads, no
   AI APIs: the entire pipeline runs offline from scikit-image/NumPy.

The price is explicitly accepted: handcrafted features do **not** capture
high-level semantics. That is why the project's architecture is retrieval
of *visually similar* images plus statistical NLP over their captions, and
why no claim of "semantic image understanding" is made anywhere.

Three independent extractors are implemented (fusion belongs to Phase 4):

| Extractor | Captures | Configuration | Dimension |
| --- | --- | --- | ---: |
| HOG | edges, contours, local shape | 9 orientations, 8×8 px cells, 2×2-cell blocks, L2-Hys, on 128×128 grayscale | **8,100** |
| Colour histogram | global colour distribution | HSV, 32 bins/channel, per-channel L1 | **96** |
| LBP | local texture | uniform, P=8, R=1, 8-bit quantised grayscale | **10** |

All three are deterministic, fixed-dimensional, NaN/inf-free, and learn
nothing from data.

## 2. HOG — Histogram of Oriented Gradients

**Input → output:** grayscale image → gradient computation → per-cell
orientation histograms → per-block normalisation → concatenated vector of
8,100 float values.

**Mathematics.** For each pixel, with centred differences:

```
Gx = I(x+1, y) - I(x-1, y)
Gy = I(x, y+1) - I(x, y-1)

magnitude    M   = sqrt(Gx² + Gy²)
orientation  θ   = atan2(Gy, Gx)   (folded into [0°, 180°) — unsigned)
```

- **Cells** (8×8 pixels): each cell votes its gradient magnitudes into a
  9-bin orientation histogram (bin width 20°). Edges and contours therefore
  appear as dominant orientations.
- **Blocks** (2×2 cells, stride 1 cell): overlapping blocks of 4 cells are
  concatenated and normalised with **L2-Hys** (L2 norm → clip values at
  0.2 → renormalise), which limits the influence of strong edges and of
  local illumination/contrast changes.
- For a 128×128 image: 16×16 cells → 15×15 blocks →
  `15 × 15 × 4 × 9 = 8,100` dimensions (verified programmatically by
  `hog_feature_dimension` and by the tests).

**Why these parameters:** 8×8-pixel cells match object-part scale after
downsampling; 9 unsigned orientation bins are the classical Dalal-Triggs
setting; L2-Hys is the scheme used in the original paper. `transform_sqrt`
is disabled so the descriptor is the plain gradient definition (easier to
explain).

**Complexity:** O(H·W) gradient work plus O(#blocks·36) normalisation —
per image, ~10 ms at 128×128.

## 3. Colour Histogram

**Input → output:** HSV image → per-channel binning → per-channel L1
normalisation → concatenation → 96 float values (each channel's 32 bins sum
to 1).

- **RGB vs HSV:** raw RGB entangles colour with brightness. HSV separates
  **H**ue (what colour), **S**aturation (colour purity) and **V**alue
  (brightness), so a red object in shadow and in sunlight moves mainly
  along V while H stays informative.
- **Binning:** 32 uniform bins per channel over [0, 1]; bin b collects the
  pixels whose channel value falls in `[b/32, (b+1)/32)`.
- **Normalisation:** per-channel L1 (`counts / pixel_count`) makes the
  vector independent of image size. Channel order in the vector:
  `[H(32), S(32), V(32)]`.
- **Complexity:** O(H·W) — a single pass over the pixels.

## 4. LBP — Local Binary Patterns

**Input → output:** 8-bit quantised grayscale → per-pixel binary
neighbourhood codes → normalised histogram → 10 float values.

For each pixel with intensity `Ic` and `P = 8` neighbours on a circle of
radius `R = 1`:

```
s(p) = 1 if I(p) >= Ic else 0
LBP  = Σ_{p=0}^{P-1} s(p) · 2^p
```

- **Uniform mapping** (`method="uniform"`): patterns with at most two 0/1
  circular transitions ("uniform" patterns — corners, edges, spots, flat
  areas) keep their own label; all remaining patterns are pooled into one
  extra label. This gives `P + 2 = 10` possible values and makes the
  histogram robust to small rotations and noise.
- **Quantisation:** the Phase 2 grayscale is rounded to 8-bit integers
  before coding. This follows the classical integer-pixel LBP definition
  and avoids the float-comparison artefacts scikit-image warns about.
- **Histogram:** counts over the 10 labels, L1-normalised (sums to 1).
  Border pixels see padded neighbours, so a perfectly constant image still
  has a strongly concentrated — not perfectly single-bin — histogram
  (documented and tested).

## 5. Feature dimensions (measured, not assumed)

```
HOG             : hog_feature_dimension((128, 128)) = 15*15 blocks * 4 cells * 9 bins = 8100
Colour histogram: 32 bins x 3 HSV channels                                          = 96
LBP             : P + 2 = 8 + 2                                                     = 10
```

Verified programmatically at extraction time, recorded in
`data/features/feature_metadata.json`, and asserted in the tests
(`EXPECTED_HOG_DIMENSION == 8100`).

## 6. Dataset-level extraction and storage

One deterministic pass over the project dataset (`extract_dataset` /
`python -m src.features.extract`):

- **8,091 images processed, 0 failed** (~153 s, ~53 img/s, single
  image-loading pass computing all three features together).
- Split counts: train 6,000 · validation 1,000 · test 1,000 · unassigned 91.
- Split-specific storage was rejected in favour of **single aligned
  matrices** (no data duplication): every row `i` of the feature matrices
  corresponds to `image_ids[i]` and `splits[i]`:

```
data/features/
├── image_ids.npy          (8091,)  <U64
├── splits.npy             (8091,)  <U16
├── hog_features.npy       (8091, 8100) float32   (~259 MB)
├── color_features.npy     (8091, 96)   float32   (  ~3 MB)
├── lbp_features.npy       (8091, 10)   float32   (  ~0.3 MB)
└── feature_metadata.json  actual configuration + counts + timing
```

- **No leakage:** extraction fits nothing and computes no dataset
  statistics; the split labels are stored so Phase 4 (fusion + PCA fit on
  training rows only) can honour the leakage rules.
- **Errors:** any unreadable image aborts the run naming the image ID —
  no silent skipping (the extraction reported `failed: 0`).

## 7. Sanity visualisation

`python -m src.features.visualize <image_filename>` renders the resized
RGB, grayscale, HOG visualisation, LBP code map and per-channel HSV
histograms to `outputs/visualizations/feature_sanity_<image>.png` (two
examples generated). Diagnostic only.

## 8. Limitations

- **No semantics.** These features encode appearance statistics only. HOG
  cannot tell *who* is in an image; the colour histogram cannot tell
  *where* a colour appears (global distribution); LBP cannot distinguish
  textures of different materials with identical local statistics. This is
  precisely why the project's later stages are framed as **visual
  similarity retrieval + statistical NLP**, not semantic understanding.
- Global colour histogram is viewpoint/background sensitive; HOG is
  translation-tolerant but not scale/rotation invariant (block-level only);
  LBP (uniform) is rotation-tolerant but discards spatial layout.
- Distorted aspect ratio from the fixed square resize (identical for all
  images, so comparisons remain consistent).
- HOG's 8,100 dimensions carry redundancy that Phase 4's PCA will exploit.

## 9. Conclusion

Phase 3 converts every project image into three explainable, deterministic
classical feature vectors (8,100 + 96 + 10 dimensions), stored aligned with
image IDs and split labels for all 8,091 images with zero failures. The
input contract for Phase 4 (fusion + normalisation + PCA on training rows
only) is established.
