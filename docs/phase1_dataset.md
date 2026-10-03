# Phase 1 — Dataset Collection and Validation

## 1. Objective

Establish a clean, validated Flickr8k dataset as the foundation for every
later phase, and produce machine-readable processed CSV files plus a
validation report. **No feature extraction, retrieval or caption generation
happens in this phase.**

Success criteria:

1. The caption file and the three official split files parse without silent
   data loss.
2. Every caption references an existing, readable image file.
3. The official splits are disjoint, duplicate-free and complete.
4. `data/processed/{captions,train,validation,test}.csv` exist, preserve the
   original caption text verbatim and are reproducible.
5. `pytest` passes (unit tests always; integration tests once the dataset is
   downloaded).

## 2. Dataset background and file formats

Flickr8k (Hodosh et al., 2013) contains ~8,000 Flickr photographs, each with
5 crowd-written reference captions collected and verified by UIUC. Multiple
references per image matter later: BLEU/ROUGE/METEOR are multi-reference
metrics, and 5 references reduce evaluation noise.

**`Flickr8k.token.txt`** — one caption per line:

```
<image_filename>#<caption_index>\t<caption text>
1000268201_693b08cb0e.jpg#0\tA child in a pink dress is climbing up a set of stairs in an entry way .
```

`caption_index` ∈ {0..4}. Captions are already lower-case-ish and
pre-tokenised (punctuation is space-separated); we do **not** normalise
them here — Phase 6 owns text preprocessing, and the evaluation in Phase 9
must compare against untouched human references.

**Split files** — `Flickr_8k.trainImages.txt`,
`Flickr_8k.devImages.txt`, `Flickr_8k.testImages.txt` list one image
filename per line. These are used as-is; the project never re-splits.

## 3. Methodology

Phase 1 is formulated as **set-theoretic integrity checking plus
descriptive statistics**. Define:

- `D` = set of image files on disk (`Images/*.jpg`)
- `C` = set of image IDs referenced by the caption file
- `S_train`, `S_val`, `S_test` = sets from the split files, `S = S_train ∪ S_val ∪ S_test`

Checks (severity in brackets):

| # | Property | Meaning | Severity |
| --- | --- | --- | --- |
| 1 | `C ⊆ D` | every caption references an existing image (`missing_images`) | error |
| 2 | readability | each file in `C` opens and passes PIL header verification (`corrupt_images`) | error |
| 3 | key uniqueness | no `image#caption_index` key appears twice (`duplicate_caption_keys`) | error |
| 4 | `S_i ∩ S_j = ∅` | splits are pairwise disjoint (`split_overlaps`) | error |
| 5 | no repeats | no image listed twice within one split file (`split_duplicates`) | error |
| 6 | `S ⊆ C` | split images have caption records (`split_images_missing_from_captions`) | error |
| 7 | `S ⊆ D` | split images exist on disk (`split_images_missing_on_disk`) | error |
| 8 | parseability | structurally broken token lines are reported, never dropped silently (`malformed_caption_lines`) | warning |
| 9 | non-empty text | captions that are whitespace-only (`empty_caption_rows`) | warning |
| 10 | `n_i = 5 ∀i` | caption count per image, `n_i = |{captions of image i}|` (`captions_per_image_violations`) | warning |
| 11 | `C \ S` | captioned images in no split — **expected** in the official release, which contains more captioned images than split-assigned images (`unassigned_images`) | warning |
| 12 | `D \ C` | stray image files without captions (`orphan_image_files`) | warning |

Descriptive statistics: `|D|`, `|C|`, total captions `N = Σ_i n_i`,
captions-per-image min/max/mean with `μ = N / |C|`, and per-split
image/caption counts. **Every reported number is computed from the files at
run time** — nothing is hard-coded from the literature.

Severity semantics: an *error* would corrupt the downstream pipeline, so
`python -m src.dataset` exits non-zero and refuses to write processed CSVs
(but still writes the report). A *warning* is an anomaly to document; the
pipeline continues.

## 4. Implementation

| Module | Responsibility |
| --- | --- |
| `src/config.py` | loads `configs/config.yaml`, resolves relative paths against the project root |
| `src/dataset/loader.py` | parses the token file (keeping malformed lines and duplicate keys visible), reads split lists, scans the image directory, checks existence/readability |
| `src/dataset/split.py` | loads the three official splits, computes duplicates/overlaps, attaches the `split` label, writes the four CSVs deterministically (sorted by `image_id, caption_id`) |
| `src/dataset/validate.py` | runs the 12 checks and packages findings with severities and counts |
| `src/dataset/report.py` | computes statistics; writes `dataset_report.json` + `dataset_report.md`; prints a console summary |
| `src/dataset/__main__.py` | orchestrates the phase; exit codes 0 / 1 (validation failed) / 2 (inputs missing) |

Image readability uses `PIL.Image.open(...).verify()`, which parses the
file header and structural markers without a full pixel decode — fast
enough for all ~8k files while still catching truncation and non-image
garbage.

## 5. How to run

```bash
cd classical-image-captioning
pip install -r requirements.txt
python -m src.dataset          # validate + build CSVs + reports
pytest                         # unit tests (+ integration once dataset present)
```

## 6. Expected output

Console (numbers computed from the actual files — project dataset after the
documented exclusion, see sections 12-13):

```
[1/6] Parsing caption file ...
      40460 caption rows, 8092 unique images, 0 malformed line(s), 0 duplicate key(s)
      project exclusions: 1 image ID(s), 5 caption row(s) removed (raw source files untouched)
[2/6] Loading official split files ...
      train=6000, validation=1000, test=1000 images
[3/6] Validating dataset ...
      [warning] unassigned_images: 91
[4/6] Writing processed CSV files ...
[5/6] Building dataset report ...
[6/6] Done.
================================================================
Original Flickr8k source         : 8092 image IDs, 40460 captions (raw files, unmodified)
Intentional exclusions           : 1 image ID(s), 5 captions (dataset.excluded_images)
Project dataset (after documented exclusion):
Images on disk                   : 8091
Images with captions             : 8091
Total captions                   : 40455
...
Validation                       : PASSED (0 errors, 91 warnings)
```

CSV format (`data/processed/*.csv`), original text preserved:

```
image_id,caption_id,caption,split
1000268201_693b08cb0e.jpg,0,A child in a pink dress is climbing up a set of stairs in an entry way .,train
```

## 7. Testing

`pytest` runs two layers:

- **Unit tests (always run)** — a synthetic 5-image mini-Flickr8k is built
  in a temp directory: parsing (verbatim text, malformed lines, duplicate
  keys, empty captions), image checks (missing files, garbage bytes,
  non-image files), split logic (overlap, duplicates, unassigned), CSV
  determinism and consistency, statistics, report round-trip, and the full
  entry point including failure modes (exit codes 1 and 2).
- **Integration tests (skip until the real dataset is downloaded)** —
  official files exist, every caption references a readable image, splits
  are disjoint and duplicate-free, and the end-to-end pipeline produces
  consistent CSVs.

## 8. Viva questions and answers

- **Why the official split instead of a fresh random one?** Comparability
  with published results, reproducibility, and no freedom to (accidentally
  or deliberately) leak test images into training.
- **Why must splits not overlap?** An image in both train and test would be
  retrieved as its own neighbour at inference — retrieval-based captioning
  would copy the reference caption and inflate BLEU. This is the data
  leakage rule for this architecture.
- **Why keep 5 captions per image?** Multi-reference evaluation (BLEU,
  METEOR) is more stable with multiple references, and more retrieval
  candidates improve later caption ranking.
- **Why preserve caption text verbatim?** The evaluation phase must compare
  generated captions against untouched human references; normalising them
  first would inflate scores.
- **What does "corrupt image" mean here?** The file cannot be opened/parsed
  by an image decoder (header verification) — it does *not* guarantee the
  pixel data is semantically meaningful.
- **Why are some images "unassigned"?** The official release captioned more
  images than it split. They are kept in `captions.csv` with
  `split=unassigned` and excluded from all three split CSVs.

## 9. Limitations

- Header verification does not fully decode pixels; rare pixel-level
  corruption can pass (Phase 2 full loading would surface it).
- Validation is structural, not semantic: a caption that does not match its
  image is indistinguishable from a correct one at this stage.
- The dataset must be downloaded manually (research licence).
- Windows/macOS filename case-insensitivity is not audited.
- The official Flickr8k distribution itself contains a captioned image with
  no image file — see Appendix (section 12). Validation therefore fails on
  the authentic dataset until a handling decision is made.

## 10. Conclusion

Phase 1 delivers a validated, officially split Flickr8k-derived corpus (one
documented upstream defect intentionally excluded, section 13) as four
deterministic CSV files plus a machine-readable validation report, with a
test suite covering both clean and deliberately broken data. This is the
sole input contract for Phase 2 (image preprocessing).

## 11. Next phase

Phase 2 — image preprocessing: loading, resizing, RGB/grayscale/HSV
conversion and normalisation, with the original image always retained.
No feature extraction yet.

## 12. Appendix — known official dataset defect

**Status: resolved (2026-10-02) — intentional project-level exclusion.**
Phase 1 validation reports exactly one error on the authentic dataset:

```
missing == ['2258277193_586949ec62.jpg.1']
```

### Finding

The image `2258277193_586949ec62.jpg` **does not exist in the official
Flickr8k image distribution.** This is a defect of the upstream dataset,
not of this project's copy or code. Evidence, all reproducible:

1. **The local image archive is intact and official.** The archive used for
   extraction (`D:\Flickr8k_Dataset.zip`) is 1,115,419,746 bytes — exactly
   the size of the canonical `Flickr8k_Dataset.zip` asset in the
   [jbrownlee/Datasets Flickr8k release](https://github.com/jbrownlee/Datasets/releases/tag/Flickr8k)
   — and passes a full CRC integrity test (`zipfile.ZipFile.testzip()`
   returns `None`).
2. **The archive contains 8,091 real images, and none matches the ID.**
   Listing the archive shows 16,182 `.jpg` entries, of which 8,092 are
   `__MACOSX/…` AppleDouble metadata twins (one per image plus one for the
   folder), leaving 8,091 real images. Zero entries contain the substring
   `2258277193` — neither as `.jpg` nor as `.jpg.1`.
3. **The local extraction is complete.** `data/raw/flickr8k/Images/`
   contains 8,091 `.jpg` files — exactly the archive's real image count.
4. **The official text files reference the missing ID.**
   `Flickr8k.token.txt` (lines 6731–6735) and `Flickr8k.lemma.token.txt`
   each carry 5 captions for `2258277193_586949ec62.jpg.1`; the ID appears
   in none of the three split files (it is one of the 92 images the official
   release never assigned to train/dev/test).
5. **This is a documented community-known quirk.** Published Flickr8k
   pipelines explicitly drop this exact ID as a known-bad data point, e.g.
   a [Kaggle ViT-BERT captioning notebook](https://www.kaggle.com/code/shirshmall/image-captioning-fine-tune-vit-bert-model-flickr8k)
   (drops `2258277193_586949ec62.jpg.1` by index), a
   [MiniCLIP write-up](https://medium.com/@ronitak33/miniclip-a-mini-implementation-of-openais-clip-model-b4fff297ea6f)
   calling it "a faulty data point in our dataset", and an
   [encoder–decoder notebook](https://github.com/priyanka-shishodia/Automating-Image-Descriptions/blob/master/encoder_decoder.ipynb)
   that deletes it because the image fails to load.

### Reproduction commands

```bash
cd "D:/ml_image_caption_generation"
.venv/Scripts/python -c "import zipfile; zf=zipfile.ZipFile('D:/Flickr8k_Dataset.zip'); \
print([n for n in zf.namelist() if '2258277193' in n]); \
print('CRC:', zf.testzip()); \
print('real images:', sum(1 for n in zf.namelist() if not n.startswith('__MACOSX') and n.lower().endswith('.jpg')))"
grep -n "2258277193" data/raw/flickr8k/Flickr8k.token.txt
grep -c "2258277193" data/raw/flickr8k/Flickr_8k.*Images.txt   # 0 in every split
ls data/raw/flickr8k/Images | grep -c "\.jpg$"                # 8091
```

### Consequences for the pipeline

- The ID is **not in train/validation/test**, so every later phase (feature
  extraction, retrieval, language model, evaluation on the official splits)
  is unaffected by it.
- The exclusion is applied at load time from configuration
  (`dataset.excluded_images`); the validator itself is unchanged and stays
  strict — a unit test (`test_pipeline_still_fails_without_exclusion`)
  proves that an unexcluded phantom ID still fails the run.

### Decision (2026-10-02)

Approved by the project owner: **remove the defective ID and its five
captions from the project dataset** via the project-level exclusion
mechanism, rather than recovering the photo. Implementation:

- `configs/config.yaml` declares `dataset.excluded_images` with the ID and
  its documented reason; `src/dataset/exclusions.py` parses and applies it.
- The raw source files are never modified (token file size/mtime verified
  unchanged).
- Every run prints the exclusion, and both reports carry dedicated
  "original source" and "intentional exclusions" sections (section 13).
- If the image file ever appears on disk for an excluded ID, the pipeline
  prints a stale-exclusion warning and flags it in the report.

### Options considered

1. **Explicit documented declaration (recommended).** Config lists the ID
   as a known official absent image with a citation to this appendix; the
   validator reports it as its own always-visible named check instead of an
   error; all 40,460 captions are preserved; the two real-dataset
   integration test assertions are updated to encode the documented quirk.
   Result: 8,091 images on disk (the official reality), PASSED with 0
   errors, nothing hidden.
2. **Keep validation absolute.** No code change; Phase 1 remains blocked on
   the upstream defect.
3. **Recover the original photo from Flickr** (photo ID `2258277193`,
   secret `586949ec62`), verify it visually against its 5 captions, and
   store it under the official ID name. Caveats: the photo may no longer be
   online, and a re-fetch is not byte-identical to the 2009 dataset file.
