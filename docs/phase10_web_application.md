# Phase 10 — Web Application (FastAPI + React)

## 1. Objective

A usable interface over the *evaluated* research system. Phase 10 changes
no algorithms: the backend calls the existing Phase 2–8 modules (the same
`CaptionGenerator` the tests exercise), and the frontend displays whatever
that pipeline actually returns — including imperfect captions, which are
never hand-corrected.

## 2. Architecture

```
User → React (Vite dev server, port 5173)
         │  same-origin /api requests (Vite proxy)
         ▼
      FastAPI (port 8000)
         │  CaptionService (artifacts loaded once at startup)
         ▼
   Phase 2–4 features → Phase 5 retrieval → Phase 6/7/8 generation
```

## 3. Backend structure

```
backend/
├── main.py                    FastAPI app: lifespan, endpoints, CORS, error handling
├── schemas.py                 Pydantic response models
├── config.py                  upload limits, parameter ranges, CORS origins
└── services/
    └── caption_service.py     CaptionService: loads artifacts once; request handling
```

`CaptionService` loads at startup: the Phase 5 retriever (6,000-image
gallery), the Phase 7 trigram (from `ngram_model.json`), unigram/bigram
(fitted once on the same 30,000 training captions), and the Phase 6
training-caption map. A per-request `CaptionGenerator` is assembled from
these shared components with the request's parameters — no training, no
recomputation of gallery features.

## 4. API

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/api/caption` | POST (multipart) | `image` file + optional `top_k`, `model`, `max_length`, `min_length` → caption JSON |
| `/api/gallery/{image_id}` | GET | one known training-gallery image (ID-validated) |
| `/api/health` | GET | `{"status": "ok"}` |
| `/api/config` | GET | safe UI configuration (no filesystem paths) |

Response example (actual values, not hard-coded):

```json
{
  "success": true,
  "generated_caption": "a man in a field",
  "language_model": "trigram",
  "top_k": 5,
  "candidate_caption_count": 25,
  "candidate_word_count": 110,
  "candidate_captions_sample": ["a child in a pink dress is climbing ...", "…"],
  "retrieved_images": [{"image_id": "432167214_c17fcc1a2d.jpg", "similarity": 0.1645}],
  "processing": {"feature_dimension": 1804, "similarity_metric": "cosine", "elapsed_ms": 253}
}
```

## 5. Validation and security

- **Uploads**: extension whitelist (`.jpg/.jpeg/.png`), content-type check,
  size limit (10 MB → HTTP 413), empty-file check, and a PIL
  open-and-verify check (HTTP 400 for non-images). Parameters are
  validated server-side (HTTP 422): `top_k` 1–10, `model` ∈
  {unigram, bigram, trigram}, `max_length` 1–30, `min_length` 0–10.
- **Path traversal**: `/api/gallery/{image_id}` accepts only IDs that are
  members of the known training gallery (no separators, membership-tested
  against the retriever's ID set) — a client can never name an arbitrary
  file. Traversal attempts return 404 (tested).
- **Errors**: processing failures return HTTP 500 with a fixed friendly
  message; stack traces go to the server log only.
- **Temporary files**: uploads are written to a private temp directory and
  deleted in a `finally` block after each request (tested: no residue).
- **CORS**: restricted to the local React dev origin(s)
  (`ALLOWED_ORIGINS` env override). Development-only configuration — a
  production deployment would pin its own origin.

## 6. Frontend structure

```
frontend/
├── package.json / vite.config.js / index.html
└── src/
    ├── main.jsx, App.jsx, api.js
    ├── components/  ImageUploader · CaptionResult · RetrievedImages ·
    │                CandidateCaptions · ProcessingInfo
    └── styles/app.css
```

Vite proxies `/api` to `http://127.0.0.1:8000` in development, so the page
uses same-origin requests. The UI shows: upload + preview (nothing is sent
before Generate), parameter controls (Top-K 1/3/5/10, model, max length),
a disabled Generate button with a loading state, the generated caption,
the retrieved training images (served through `/api/gallery/...`) with
similarity scores, the candidate captions and counts, processing details,
a fixed methodology panel, and the standing disclaimer that the
descriptors are low-level and captions may be generic or imperfect.

## 7. Running

```powershell
# Backend (from the project root)
.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --reload        # http://127.0.0.1:8000/docs

# Frontend (second terminal)
cd frontend
npm install
npm run dev                                        # http://localhost:5173
```

## 8. End-to-end verification (measured)

| Query | Route | Result |
| --- | --- | --- |
| `1000268201_693b08cb0e.jpg` (training image, uploaded as a copy) | backend :8000 | 200; caption "a man in a field"; 25 candidates; rank-1 = the same image at similarity ≈ 1.0 — expected for an uploaded *copy* of a gallery image (self-exclusion works by dataset filename, which an upload does not carry) |
| `1001773457_577c3a7d70.jpg` via frontend proxy | :5173 → :8000 | 200; bigram, K=3; caption "a man is a man is on the dog" (repetition caps hold: every word ≤ 2) |
| external PNG (synthetic sky/field/sun scene, not in Flickr8k) | :5173 | 200; caption "a man in a blue and white"; retrieved neighbours are blue/white-toned scenes — colour-driven retrieval, exactly as documented |
| Swagger UI | :8000/docs | 200 |

Error paths verified by tests: non-image bytes (400), unsupported
extension (400), wrong content type (400), empty file (400), oversized
(413), missing field (422), invalid model/parameters (422), unknown
gallery ID and traversal attempts (404).

## 9. Testing

21 backend tests (`tests/test_api.py`, FastAPI `TestClient`, real
artifacts): health, config, full caption flow, exact response schema,
gallery membership, defaults and overrides, model/parameter validation,
upload validation, 413/422/400/404 paths, path-traversal attempts, and
temp-file cleanup. Full suite: **237 passed** (216 prior + 21 new).

## 10. Limitations

- Single-process in-memory service (gallery + models held in RAM,
  ~350 MB); not a horizontally scalable deployment.
- Development-oriented CORS and Vite proxy; production would serve the
  built frontend and pin origins.
- No authentication, no rate limiting, no upload persistence — out of
  scope for an academic demonstration.
- Uploaded copies of *training* images self-match at rank 1 (filename-
  based self-exclusion cannot recognise renamed copies).
- The UI displays whatever the evaluated system produces, including its
  generic or incorrect captions — by design (Phase 9 conclusions are not
  cosmetically altered).
