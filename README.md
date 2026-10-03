# Classical ML and Vision-Language API Based Image Captioning System

**Two clearly-labelled captioning methods for one uploaded image, side by
side in one web application.**

**Classical Pipeline** — the project's own fully local, explainable
baseline (Phases 1–9), built **entirely** from classical techniques
(hand-crafted visual features, PCA, retrieval, n-gram language models).
Its core algorithms contain no LLMs, Transformers, pretrained CNNs or
external AI APIs.

```text
Classical Pipeline
HOG + HSV Color Histogram + LBP
        ↓
Feature Fusion + PCA (fitted on training data only)
        ↓
Cosine Similarity Retrieval (6,000-image training gallery)
        ↓
N-gram Language Model (trigram, Laplace smoothing)
        ↓
Classical Caption
```

**Vision API Pipeline** — an external hosted model (OpenAI Responses API),
added in Phase 10 as a comparison method. It is **not** part of the
classical ML baseline and was **not** trained by this project.

```text
Vision API Pipeline
Uploaded Image
        ↓
FastAPI backend (API key stays server-side)
        ↓
Vision-Language API
        ↓
Natural Language Caption
```

Both captions are returned together by `/api/caption` and displayed side
by side, exactly as produced — no editing, no scoring, no declared winner.
The classical branch performs *visual similarity*, not semantic
understanding; measured on the test set (Phase 9): BLEU-1 0.488 · BLEU-4
0.090 · ROUGE-L 0.343. The vision-API caption typically describes the
actual objects and setting — the comparison makes the gap concrete and
visible.

## Project status

| Phase | Scope | Status |
| --- | --- | --- |
| 1 | Dataset collection and validation | **COMPLETE** — with one documented upstream dataset exclusion ([docs/phase1_dataset.md](docs/phase1_dataset.md), §12–13) |
| 2 | Image preprocessing | **COMPLETE** |
| 3 | Classical feature extraction (HOG / Color / LBP) | **COMPLETE** |
| 4 | Feature fusion and PCA | **COMPLETE** — 8,206-D fused → 1,804-D at 95.00 % variance, fitted on train only |
| 5 | Image retrieval (cosine, training gallery) | **COMPLETE** — 6,000-image gallery, deterministic Top-K |
| 6 | Caption preprocessing | **COMPLETE** — 40,000 captions tokenized; 7,374-token vocabulary from train only |
| 7 | N-gram language model | **COMPLETE** — trigram (132,845 trigrams), Laplace α=1.0, fitted on train only |
| 8 | Caption generation | **COMPLETE** — retrieval evidence + greedy trigram decoding, deterministic |
| 9 | Evaluation (BLEU/ROUGE + experiments) | **COMPLETE** — test BLEU-1 0.488 / BLEU-4 0.090 / ROUGE-L 0.343; leakage checks PASS |
| 10 | Web application (FastAPI + React) | **COMPLETE** — upload image → caption + retrieval + evidence |

Per-phase documentation lives in [`docs/`](docs/):
[phase1_dataset.md](docs/phase1_dataset.md),
[phase2_preprocessing.md](docs/phase2_preprocessing.md),
[phase3_features.md](docs/phase3_features.md),
[phase4_feature_fusion_pca.md](docs/phase4_feature_fusion_pca.md),
[phase5_image_retrieval.md](docs/phase5_image_retrieval.md),
[phase6_caption_preprocessing.md](docs/phase6_caption_preprocessing.md),
[phase7_ngram_language_model.md](docs/phase7_ngram_language_model.md),
[phase8_caption_generation.md](docs/phase8_caption_generation.md),
[phase9_evaluation.md](docs/phase9_evaluation.md),
[phase10_web_application.md](docs/phase10_web_application.md).

## Requirements

- Python 3.10+ (developed on 3.13)
- pandas, Pillow, PyYAML, pytest (Phase 1)
- NumPy, scikit-image, tqdm (Phases 2–3)
- matplotlib (Phase 3 diagnostic visualization only)

## Setup

```bash
cd ml_image_caption_generation
python -m venv .venv
source .venv/Scripts/activate      # Git Bash on Windows
# .venv\Scripts\activate           # CMD on Windows
# .venv/bin/activate               # Linux / macOS
pip install -r requirements.txt
```

## Dataset acquisition

The project uses **Flickr8k** (~8k images, 5 human captions each) with the
**official** train/validation/test split files. The dataset is not bundled
with the repository (research licence); download it once and place the five
items below under `data/raw/flickr8k/`:

```
data/raw/flickr8k/
├── Images/                      # the .jpg files
├── Flickr8k.token.txt           # 5 captions per image
├── Flickr_8k.trainImages.txt    # official train image list
├── Flickr_8k.devImages.txt      # official validation image list
└── Flickr_8k.testImages.txt     # official test image list
```

Options, in order of preference:

1. **Official source (UIUC)** — the Flickr8k homepage of Julia
   Hockenmaier's group: <https://shannon.cs.illinois.edu/DenotationGraph>.
   Fill in the request form; download links are emailed to you
   (free for research).
2. **Direct mirror (GitHub releases)** — download both archives and unzip:
   - `Flickr8k_Dataset.zip` (1.04 GB, the images):
     <https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip>
   - `Flickr8k_text.zip` (2.2 MB, token + split files):
     <https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip>
3. **Kaggle mirrors** — e.g. *original-flickr8k-dataset* or *adityajn105/flickr8k*.
   Caution: some mirrors ship a differently formatted `captions.txt` and no
   official split files; this project expects the classic five files above.

### Dataset exclusions (intentional, documented)

The official Flickr8k release references one image ID —
`2258277193_586949ec62.jpg.1` — that is absent from the official image
archive (a known upstream defect; full evidence in
[`docs/phase1_dataset.md`](docs/phase1_dataset.md), §12). The project
excludes this ID and its 5 captions via `dataset.excluded_images` in
`configs/config.yaml`; the raw source files are never modified. The project
dataset is therefore a documented derivative of Flickr8k —
**8,091 images / 40,455 captions** — not the complete original dataset
(8,092 images / 40,460 captions).

## Usage

```bash
# Phase 1 — validate the dataset, build processed CSVs and the report
python -m src.dataset

# Phase 2 + 3 — extract HOG, color histogram and LBP features for the
# whole project dataset in a single pass (writes data/features/*.npy)
python -m src.features.extract
python -m src.features.extract --limit 50          # quick debug run
python -m src.features.extract --splits train      # subset of splits

# Phase 4 — standardize (train-only), fuse to 8206-D, PCA to 95% variance
# (writes data/features/fusion/*)
python -m src.features.fusion

# Phase 5 — retrieve the Top-K most visually similar training images
python -m src.retrieval.run_retrieval --image 1000268201_693b08cb0e.jpg --top-k 5

# Phase 6 — caption preprocessing: normalize, tokenize, training-only
# vocabulary, integer encoding (writes data/captions/*)
python -m src.captions.pipeline

# Phase 7 — train the trigram language model on training captions only
# (writes data/language_model/*; validation used for perplexity only)
python -m src.language_model.train

# Phase 8 — generate a caption for one image (retrieval + greedy decoding)
python -m src.caption_generation.run --image 1000268201_693b08cb0e.jpg
python -m src.caption_generation.run --image 1000268201_693b08cb0e.jpg --top-k 10 --max-length 15

# Phase 9 — experimental evaluation (validation = experiments, test = final run)
python -m src.evaluation.run --split validation --all-models
python -m src.evaluation.run --split test

## Running the Application

### Vision API setup (one-time, optional but recommended)

Copy `.env.example` to `.env` and add your key — it stays on the backend
and is never exposed to the frontend:

```powershell
Copy-Item .env.example .env
# then edit .env:  OPENAI_API_KEY=sk-...
```

Without a key the app runs fully — the UI shows
"Vision API: Not configured" and the classical caption alone.

### Backend

```powershell
.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --reload
```

API + Swagger UI: <http://127.0.0.1:8000/docs>

### Frontend (second terminal)

```powershell
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173> (the Vite dev server proxies `/api` to the
backend on port 8000).

# Phase 3/4/5 — diagnostic visualizations (no deep learning)
python -m src.features.visualize 1000268201_693b08cb0e.jpg
python -m src.features.visualize_pca
python -m src.retrieval.visualize --image 1000268201_693b08cb0e.jpg --top-k 5

# Run the test suite (237 tests; dataset-dependent tests skip when absent)
pytest
```

Outputs:

- `data/processed/captions.csv`, `train.csv`, `validation.csv`, `test.csv`
  — captions with split labels (Phase 1)
- `data/processed/dataset_report.json` / `dataset_report.md` — validation report
- `data/features/image_ids.npy`, `splits.npy`, `hog_features.npy`,
  `color_features.npy`, `lbp_features.npy`, `feature_metadata.json`
  (Phase 3; all rows aligned)
- `data/features/fusion/` — per-split PCA features, fitted scalers, PCA,
  metadata (Phase 4)
- `data/captions/` — processed caption CSVs, `vocabulary.json`,
  `encoded_captions.json`, `caption_metadata.json` (Phase 6)
- `outputs/visualizations/` — diagnostic figures (Phases 3–5)

Feature configuration (all values also recorded in `feature_metadata.json`):

| Feature | Configuration | Dimension |
| --- | --- | ---: |
| HOG | 9 orientations, 8×8 px cells, 2×2 cell blocks, L2-Hys, on 128×128 grayscale | 8,100 |
| Color histogram | HSV, 32 bins/channel, per-channel L1 normalization | 96 |
| LBP | uniform, P=8, R=1, on 8-bit quantized grayscale | 10 |
| **Fused (Phase 4)** | per-family StandardScaler (train-only fit) + concatenation | 8,206 |
| **PCA (Phase 4)** | 95 % variance target, exact full-SVD solver, train-only fit | 1,804 |
| **Retrieval (Phase 5)** | cosine similarity over the 6,000-image training gallery, deterministic Top-K | — |
| **Captions (Phase 6)** | lowercase → punctuation→space → whitespace collapse → split; `<START>`/`<END>` boundaries; vocabulary from train only (`min_frequency=1`) | 7,374 tokens |
| **Language model (Phase 7)** | trigram MLE + Laplace (add-α, default 1.0), log-space scoring; fitted on 30,000 training captions only | 7,372 / 58,589 / 132,845 n-grams |
| **Generation (Phase 8)** | candidates from Top-5 retrieved training captions (≤ 25), score = trigram log-prob + retrieval-evidence bonus, greedy decoding; K=5, max 20 words, min 3 | ≤ 5·K captions |
| **Evaluation (Phase 9)** | corpus BLEU-1..4 (multi-reference, cumulative) + ROUGE-1/2/L (per-image max-F1); validation for experiments, one-shot test run | test: BLEU-1 0.488 · BLEU-4 0.090 · ROUGE-L 0.343 |

## Project layout

```
ml_image_caption_generation/
├── README.md, requirements.txt, .gitignore, LICENSE, pytest.ini
├── configs/config.yaml          # paths, exclusions, knobs
├── data/raw/flickr8k/           # dataset goes here (not committed)
├── data/processed/              # Phase 1 CSVs + reports
├── data/features/               # Phase 3 matrices + Phase 4 fusion/ (not committed)
├── data/captions/               # Phase 6 processed captions + vocabulary (not committed)
├── data/language_model/         # Phase 7 n-gram model + metadata (not committed)
├── data/evaluation/             # Phase 9 predictions, metrics, analyses (not committed)
├── docs/                        # per-phase documentation
├── outputs/visualizations/      # diagnostic figures (not committed)
├── backend/                     # Phase 10: FastAPI app + caption service
├── frontend/                    # Phase 10: React (Vite) UI
├── src/
│   ├── config.py                # config loading / path resolution
│   ├── dataset/                 # Phase 1: parse, validate, split, report
│   ├── preprocessing/           # Phase 2: load, resize, grayscale, HSV
│   ├── features/                # Phase 3: hog, color_histogram, lbp,
│   │                            #          extract (CLI), visualize (CLI)
│   │                            # Phase 4: fusion (CLI), visualize_pca (CLI)
│   ├── retrieval/               # Phase 5: retriever, run_retrieval (CLI),
│   │                            #          visualize (CLI)
│   ├── captions/                # Phase 6: preprocess, vocabulary,
│   │                            #          pipeline (CLI)
│   ├── language_model/          # Phase 7: ngram, smoothing, evaluate,
│   │                            #          train (CLI)
│   ├── caption_generation/      # Phase 8: candidates, generator,
│   │                            #          run (CLI)
│   └── evaluation/              # Phase 9: caption_metrics, retrieval_metrics,
│                                #          experiment, comparison, error_analysis,
│                                #          run (CLI)
└── tests/                       # pytest suite (unit + integration + API)
```

The project is complete through Phase 10: dataset → visual branch →
caption branch → evaluation → web application.

## Data-leakage policy

- Phase 3 feature extraction is deterministic and fits **nothing**.
- Phase 4 fits three per-family `StandardScaler`s and the PCA **exclusively
  on the 6,000 training rows** (`n_samples_seen_ == 6000` is asserted by
  tests on the saved artifacts); validation, test and the 91 unassigned
  images are only ever transformed, and unassigned rows are stored
  separately.
- Feature rows carry their split label (`splits.npy`), so Phase 5 will fit
  the retrieval index and K-Means on **training** data only.
- Validation data will be used for hyperparameter selection only; test
  data only once, for final evaluation.
