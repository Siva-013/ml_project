"""Phase 10 — FastAPI application for the classical caption generator +
external vision-language API comparison.

Endpoints:

* ``POST /api/caption``       — multipart upload → BOTH captions:
  the Phase 2–8 classical pipeline result and (when configured) the OpenAI
  vision-language API caption for the same image. The API is optional: if
  its key is missing or it fails, the classical result is still returned
  with a short, safe error in the ``vision_api`` block.
* ``GET  /api/gallery/{id}``  — training-gallery image (ID-validated, no
  arbitrary filesystem access)
* ``GET  /api/health``        — liveness
* ``GET  /api/config``        — safe configuration for the UI, including
  whether the vision API is configured (never the key)

Artifacts (retriever gallery, language models, training captions) are
loaded once at startup and reused; uploads are written to a private temp
directory (classical pipeline) and deleted after processing. Errors are
returned as clean HTTP codes with short messages — no stack traces.
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

# Make the project root importable when uvicorn is started from it.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

from backend import config as backend_config
from backend.schemas import (
    CaptionResponse,
    ClassicalResult,
    ConfigResponse,
    HealthResponse,
    VisionAPIResult,
    VisionStatus,
)
from backend.services import vision_api
from backend.services.caption_service import CaptionService

MODEL_LABELS = {
    "unigram": "Unigram N-gram",
    "bigram": "Bigram N-gram",
    "trigram": "Trigram N-gram",
}

SERVICE: CaptionService | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global SERVICE
    vision_api.configure(
        model=backend_config.VISION_MODEL,
        api_key=backend_config.VISION_API_KEY,
        timeout_seconds=backend_config.VISION_TIMEOUT_SECONDS,
    )
    SERVICE = CaptionService()
    yield
    SERVICE.shutdown()


app = FastAPI(
    title="Classical ML + Vision-Language API Image Captioning",
    description="Two independent captioning methods for one uploaded image: "
    "a classical pipeline (HOG + color histogram + LBP → fusion + PCA → "
    "cosine retrieval → trigram language model) and the OpenAI "
    "vision-language API — clearly labelled, side by side.",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=backend_config.ALLOWED_ORIGINS,  # local dev servers only
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _validate_upload(file: UploadFile, data: bytes) -> tuple[str, str]:
    """Validate extension, content type, size and decodability of an upload.

    Returns:
        ``(safe_suffix, mime_type)`` for the temporary file / vision call.

    Raises:
        HTTPException: 413 for oversized files, 400 for anything else that
            makes the file unusable.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in backend_config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {suffix or '(none)'}; allowed: "
            f"{sorted(backend_config.ALLOWED_EXTENSIONS)}",
        )
    if file.content_type and file.content_type.lower() not in backend_config.ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type '{file.content_type}'",
        )
    max_bytes = backend_config.MAX_UPLOAD_MB * 1024 * 1024
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large; the limit is {backend_config.MAX_UPLOAD_MB} MB",
        )
    mime_type = (file.content_type or "image/jpeg").lower()
    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image")
    return suffix, mime_type


def _service() -> CaptionService:
    if SERVICE is None:
        raise HTTPException(status_code=503, detail="Service is starting up")
    return SERVICE


@app.post("/api/caption", response_model=CaptionResponse)
async def generate_caption(
    image: UploadFile = File(...),
    top_k: int = Form(backend_config.DEFAULT_TOP_K),
    model: str = Form(backend_config.DEFAULT_MODEL),
    max_length: int = Form(backend_config.DEFAULT_MAX_LENGTH),
    min_length: int = Form(backend_config.DEFAULT_MIN_LENGTH),
) -> CaptionResponse:
    """Run both captioning methods on the same uploaded image."""
    data = await image.read()
    suffix, mime_type = _validate_upload(image, data)
    try:
        backend_config.validate_parameters(top_k, model, max_length, min_length)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))

    service = _service()

    # ---- classical pipeline (Phase 2–8) --------------------------------
    upload_path = service.save_upload(data, suffix)
    try:
        try:
            classical = service.generate(
                upload_path,
                top_k=top_k,
                model=model,
                max_length=max_length,
                min_length=min_length,
            )
        except Exception as error:  # noqa: BLE001 — never leak stack traces
            print(f"[caption] classical pipeline failed: {error}", file=sys.stderr)
            raise HTTPException(
                status_code=500,
                detail="Unable to generate a caption for this image. Please try another image.",
            )
    finally:
        service.cleanup(upload_path)

    # ---- external vision-language API (optional, backend-only) ---------
    vision_result = VisionAPIResult(
        caption=None,
        provider=vision_api.provider(),
        model=vision_api.model_name(),
        error=None,
    )
    if vision_api.is_configured():
        try:
            # the same uploaded image bytes go to the API (in memory)
            vision_result.caption = vision_api.caption_image(data, mime_type)
        except vision_api.VisionAPIError as error:
            vision_result.error = str(error)
    else:
        vision_result.error = "Vision API not configured"

    return CaptionResponse(
        success=True,
        classical=ClassicalResult(
            caption=classical["generated_caption"],
            model=model,
            model_label=MODEL_LABELS[model],
            top_k=top_k,
            candidate_caption_count=classical["candidate_caption_count"],
            candidate_word_count=classical["candidate_word_count"],
            candidate_captions_sample=classical["candidate_captions_sample"],
            retrieved_images=classical["retrieved_images"],
            processing=classical["processing"],
        ),
        vision_api=vision_result,
    )


@app.get("/api/gallery/{image_id}")
async def gallery_image(image_id: str) -> FileResponse:
    """Serve one known training-gallery image (no arbitrary paths)."""
    service = _service()
    try:
        path = service.gallery_image_path(image_id)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error))
    return FileResponse(path, media_type="image/jpeg", filename=image_id)


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@app.get("/api/config", response_model=ConfigResponse)
async def api_config() -> ConfigResponse:
    """Safe configuration for the UI (no filesystem paths, no API key)."""
    return ConfigResponse(
        default_top_k=backend_config.DEFAULT_TOP_K,
        default_model=backend_config.DEFAULT_MODEL,
        supported_models=list(backend_config.ALLOWED_MODELS),
        allowed_top_k_range=[backend_config.MIN_TOP_K, backend_config.MAX_TOP_K],
        max_upload_mb=backend_config.MAX_UPLOAD_MB,
        allowed_extensions=sorted(backend_config.ALLOWED_EXTENSIONS),
        vision_api=VisionStatus(
            configured=vision_api.is_configured(),
            provider=vision_api.provider(),
            model=vision_api.model_name(),
        ),
    )
