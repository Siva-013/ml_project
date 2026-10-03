"""Phase 10 — FastAPI application for the classical caption generator.

Endpoints:

* ``POST /api/caption``       — multipart upload → pipeline → caption JSON
* ``GET  /api/gallery/{id}``  — training-gallery image (ID-validated, no
  arbitrary filesystem access)
* ``GET  /api/health``        — liveness
* ``GET  /api/config``        — safe configuration for the UI

Artifacts (retriever gallery, language models, training captions) are
loaded once at startup and reused for every request; uploads are written
to a private temporary directory and deleted after processing. Errors are
returned as clean HTTP codes with short messages — no stack traces.
"""

from __future__ import annotations

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
from backend.schemas import CaptionResponse, ConfigResponse, HealthResponse
from backend.services.caption_service import CaptionService

SERVICE: CaptionService | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global SERVICE
    SERVICE = CaptionService()
    yield
    SERVICE.shutdown()


app = FastAPI(
    title="Classical ML Image Caption Generator",
    description="Classical computer vision + statistical NLP caption generation "
    "(HOG + color histogram + LBP → fusion + PCA → cosine retrieval → "
    "trigram language model). No LLMs, no neural networks.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=backend_config.ALLOWED_ORIGINS,  # local dev servers only
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _validate_upload(file: UploadFile, data: bytes) -> str:
    """Validate extension, content type, size and decodability of an upload.

    Returns:
        The safe suffix to use for the temporary file.

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
    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image")
    return suffix


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
    """Generate a caption for an uploaded image (Phase 2–8 pipeline)."""
    data = await image.read()
    suffix = _validate_upload(image, data)
    try:
        backend_config.validate_parameters(top_k, model, max_length, min_length)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))

    service = _service()
    upload_path = service.save_upload(data, suffix)
    try:
        try:
            response = service.generate(
                upload_path,
                top_k=top_k,
                model=model,
                max_length=max_length,
                min_length=min_length,
            )
        except Exception as error:  # noqa: BLE001 — never leak stack traces
            print(f"[caption] processing failed: {error}", file=sys.stderr)
            raise HTTPException(
                status_code=500,
                detail="Unable to generate a caption for this image. Please try another image.",
            )
    finally:
        service.cleanup(upload_path)
    return CaptionResponse(**response)


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
    """Safe configuration for the UI (no filesystem paths)."""
    return ConfigResponse(
        default_top_k=backend_config.DEFAULT_TOP_K,
        default_model=backend_config.DEFAULT_MODEL,
        supported_models=list(backend_config.ALLOWED_MODELS),
        allowed_top_k_range=[backend_config.MIN_TOP_K, backend_config.MAX_TOP_K],
        max_upload_mb=backend_config.MAX_UPLOAD_MB,
        allowed_extensions=sorted(backend_config.ALLOWED_EXTENSIONS),
    )
