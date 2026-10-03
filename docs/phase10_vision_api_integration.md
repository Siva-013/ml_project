# Phase 10 (revision) — Vision-Language API Integration

## 1. Why the API was added

Phases 1–9 measured the classical pipeline honestly: BLEU-4 ≈ 0.09,
ROUGE-L ≈ 0.34 on test, with failure analysis showing generic captions
("a man in a …") and colour/texture-driven retrieval. The root cause is
that HOG, HSV colour histograms and LBP are **low-level statistical
descriptors** — they encode edges, colour distributions and texture, and
have no mechanism for object identity, actions or scene semantics. No
amount of engineering on top of them changes that.

The project therefore adds a second, independent captioning method: a
hosted **vision-language API** (OpenAI Responses API with an
image-capable model) that receives the *same uploaded image* and returns a
caption. This enables a factual side-by-side comparison in the running
application: the classical baseline next to a modern hosted model, each
clearly labelled. **The external API is not part of the classical ML
baseline and was not trained by this project** — it is a separate modern
component behind a clearly marked UI panel.

## 2. What changed — and what did not

Unchanged (verified by the full test suite, 241 passing): HOG, colour
histogram, LBP, standardization + fusion + PCA, cosine retrieval, the
n-gram models, caption preprocessing, the Phase 8 generator (its decode
loop was already a shared `greedy_decode` function), Phase 9 artifacts and
conclusions. No model was retrained; the API is strictly additive.

New: the `/api/caption` response now carries **both** results. The
following is the **actually observed** response for the synthetic
sky/field test image *without* a configured key (graceful degradation):

```json
{
  "success": true,
  "classical": {
    "caption": "a man in a blue and white",
    "model": "trigram", "model_label": "Trigram N-gram", "top_k": 5,
    "candidate_caption_count": 25, "candidate_word_count": 110,
    "candidate_captions_sample": ["…", "…"],
    "retrieved_images": [{"image_id": "3284955091_59317073f0.jpg", "similarity": 0.2101}],
    "processing": {"feature_dimension": 1804, "similarity_metric": "cosine", "elapsed_ms": 253}
  },
  "vision_api": {
    "caption": null,
    "provider": "OpenAI",
    "model": "gpt-4o-mini",
    "error": "Vision API not configured"
  }
}
```

With a key present in `.env`, `vision_api.caption` holds the verbatim
text returned by the API for that image (mocked-response tests verify this
path; a real call additionally requires the user's paid key). When the API
is unavailable, `vision_api.caption` is `null` and a short `error` is set —
the classical result is always returned when possible.

## 3. Request flow

```
React (upload) → POST /api/caption (multipart) → FastAPI
    ├─ validate file (extension/content-type/size/PIL-verify)
    ├─ classical: Phase 2 → 3 → 4 → 5 → candidates → trigram (greedy)
    └─ vision:    the same image bytes → base64 data URL
                  → OpenAI Responses API (input_image + prompt)
                  → response.output_text
← JSON {classical, vision_api}
```

The API call happens **only on the backend**. The frontend never sees or
sends the key.

## 4. Vision prompt (verbatim)

```
Generate one concise, factual caption describing the main visible content of this image.

Mention the main subjects, their important action, and the relevant setting when clearly visible.

Do not invent objects, people, actions, locations, or relationships that are not visually supported.

Return only the caption.
```

The response text is displayed verbatim — no post-processing into a
predetermined caption.

## 5. Environment variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `OPENAI_API_KEY` | authenticates the vision call | empty → vision disabled, UI shows "Not configured" |
| `OPENAI_VISION_MODEL` | image-capable model name | `gpt-4o-mini` |
| `VISION_TIMEOUT_SECONDS` | client timeout | `30` |

Setup: copy `.env.example` to `.env`, paste your key after
`OPENAI_API_KEY=`, restart the backend. `.env` is gitignored (`.env.example`
is the committed template); `test_env_file_is_gitignored` asserts the
ignore rule.

## 6. Security considerations

- The key exists only in the backend process: never in React/JS/HTML,
  never in API responses (tested: `/api/config` and caption responses
  contain no key material), never logged.
- The frontend learns only `configured: true/false` plus the model name
  (`GET /api/config` → `vision_api` block) — that powers the
  "Vision API: Connected / Not configured" indicator.
- All Phase 10 upload/security rules remain: extension/content-type/
  size/PIL validation, parameter range validation, ID-validated gallery
  route (no arbitrary paths), private temp files deleted per request.
- Automated tests **never make real paid calls**: the default environment
  has no key, and the vision function is monkeypatched wherever the
  "configured" path is exercised.

## 7. Frontend behaviour

The page shows a status pill ("Vision API: Connected / Not configured"
with provider and model name). After Generate, the results appear as a
side-by-side comparison: **Classical ML System** (caption + model label +
Top-K + candidate counts) and **Vision-Language API** (caption, or the
safe error message) — no scores, no winner, factual presentation. Below:
the retrieved training images with cosine similarities, the candidate
captions, processing details, and the standing methodology/disclaimer
panel. The caption texts are rendered exactly as returned.

## 8. Error handling matrix

| Condition | Behaviour |
| --- | --- |
| Missing/invalid/oversized/malformed image | HTTP 400/413/422 before any pipeline runs |
| `OPENAI_API_KEY` missing | HTTP 200; `vision_api.error = "Vision API not configured"`; classical intact |
| Authentication failure (401) | `vision_api.error = "Vision API authentication failed"` |
| Rate limit (429) | `vision_api.error = "Vision API rate limit reached"` |
| Timeout / network error | `"Vision API timeout"` / `"Vision API network error"` |
| Unexpected API response (empty output) | `"Vision API returned an unexpected response"` |
| Classical pipeline failure | HTTP 500 with a fixed friendly message; stack trace only in server logs |

## 9. Testing

Backend (`tests/test_api.py`, 25 tests): all previous upload/security/
gallery/temp-cleanup tests updated to the nested schema, plus vision
tests — mocked success (caption returned verbatim), authentication
failure, rate limit, unconfigured status (deterministic even when a real
key exists later), key never in responses, `.env` gitignored. **No test
makes a real paid API call.**

Frontend (`frontend/src/__tests__/app.test.jsx`, 6 tests, vitest +
Testing Library, API module mocked): page render + Connected status,
image selection + preview, loading state with disabled button, both
captions rendered with clear labels, vision-error rendering with the
classical caption intact, and friendly generation-failure rendering.

Full verification: **241 Python tests passed · 6 frontend tests passed.**

## 10. Running the complete application

```powershell
# one-time: put your key in .env (see section 5)

# terminal 1 — backend
cd D:\ml_image_caption_generation
.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --reload          # http://127.0.0.1:8000/docs

# terminal 2 — frontend
cd D:\ml_image_caption_generation\frontend
npm install
npm run dev                                          # http://localhost:5173
```

Browser workflow: open the app → choose an image (preview appears) →
Generate Caption → the comparison panel shows the classical caption and
the vision caption side by side, with retrieved images + similarity
scores below.

## 11. Difference between the two captions (measured expectation)

The classical caption is assembled from the vocabulary of visually similar
training images by a Laplace-smoothed trigram — locally plausible,
frequently generic, semantically loose (Phase 9 measured test BLEU-1
0.488 / BLEU-4 0.090). The vision-language caption is produced by a
large hosted multimodal model and typically describes the actual objects,
actions and setting. They are different *kinds* of system: one is the
project's fully local, explainable classical baseline; the other is an
external commercial service shown for comparison. Neither output is
edited by the application.

## 12. Limitations

- A real vision call requires the user's own paid OpenAI key; without it
  the UI honestly reports "Not configured".
- The external API adds network latency (seconds) and per-call cost.
- External hosted models can still hallucinate; the prompt restricts but
  cannot eliminate that.
- No evaluation metrics are computed for the vision API in this phase —
  any metric comparison would be future work (and Phase 9's classical
  numbers remain unchanged).
