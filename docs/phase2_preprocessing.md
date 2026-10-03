# Phase 2 — Image Preprocessing

> Implemented together with Phase 3 (2026-10-02): Phase 3 consumes these
> outputs directly, so the module was built as its exact prerequisite.

## 1. Objective

Turn every raw Flickr8k JPEG into a fixed-size, numerically normalised set
of representations that the Phase 3 feature extractors consume:

| Output | Shape | Range | Purpose |
| --- | --- | --- | --- |
| `rgb` | (128, 128, 3) float32 | [0, 1] | display, HSV conversion |
| `grayscale` | (128, 128) float32 | [0, 1] | HOG, LBP |
| `hsv` | (128, 128, 3) float32 | [0, 1] | color histogram |

Preprocessing is **deterministic**: no learned parameters, no randomness.
The original files on disk are never modified.

## 2. Methodology

1. **Loading** — `PIL.Image.open(...).convert("RGB")`. Every Flickr8k image
   is a JPEG (some non-RGB encodings are normalised away here). A file that
   cannot be decoded raises a meaningful `ValueError`; nothing is skipped
   silently (Phase 1 has already validated the dataset, so a decode failure
   at this stage is a data-integrity problem).
2. **Resizing** — to a fixed **128×128** with **Lanczos** resampling.
   Rationale: with the Phase 3 HOG configuration (9 orientations, 8×8-px
   cells, 2×2-cell blocks) 128×128 yields an 8,100-dimensional HOG vector
   (~262 MB float32 for 8,091 images), whereas 256×256 would yield 34,596
   dimensions (~1.1 GB) before PCA without adding useful information for
   handcrafted features. Lanczos keeps edges sharp for gradient features.
3. **Grayscale** — Rec. 709 luma via `skimage.color.rgb2gray`:
   `Y = 0.2125·R + 0.7154·G + 0.0721·B`.
4. **HSV** — `skimage.color.rgb2hsv` on the `[0, 1]` RGB array; channel
   order H, S, V, each in `[0, 1]`.
5. **Normalisation** — all arrays are float32 in `[0, 1]`
   (`uint8 / 255`), so every downstream extractor receives identical,
   resolution-independent inputs.

## 3. Implementation

- `src/preprocessing/image_preprocessing.py`
  - `TARGET_SIZE = (128, 128)`, `RESIZE_FILTER = LANCZOS` — the only knobs.
  - `load_rgb(path)` — RGB loading with meaningful errors.
  - `resize_image(image, size)` — Lanczos resize.
  - `image_to_array(image)` — float32 `[0, 1]` conversion.
  - `to_grayscale(rgb)` / `to_hsv(rgb)` — colour-space conversions with
    input validation.
  - `preprocess_image(path, image_id=None)` — the single entry point used
    by Phase 3; returns a frozen `PreprocessedImage`.
- Config: no separate knobs in `configs/config.yaml`; the target size is a
  documented constant because the HOG dimensionality depends on it.

## 4. Testing

`tests/test_preprocessing.py` (21 tests): loading (valid, missing,
undecodable), resizing (size, invalid size), array conversion (dtype,
shape, range), grayscale (shape, range, exact luma of pure red, rejection
of RGB/empty input), HSV (pure red → H=0, S=1, V=1), the composed entry
point (determinism, aligned shapes), and one real-image integration test.

## 5. Limitations

- Fixed square resize distorts aspect ratio (all images are distorted
  identically, so retrieval remains consistent; classical pipelines often
  accept this trade-off).
- Downsampling to 128×128 discards fine detail — acceptable for shape,
  colour-distribution and texture statistics, but a limitation to state in
  the report.
- Grayscale discards colour; that is why colour information enters through
  the separate HSV histogram in Phase 3.

## 6. Conclusion

Phase 2 provides one deterministic, validated preprocessing path for every
image in the project dataset and is the sole input contract for Phase 3
feature extraction.
