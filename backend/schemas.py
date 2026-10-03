"""Phase 10 — Pydantic response schemas for the caption API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievedImage(BaseModel):
    image_id: str
    similarity: float


class ProcessingInfo(BaseModel):
    feature_dimension: int
    similarity_metric: str = "cosine"
    elapsed_ms: int


class CaptionResponse(BaseModel):
    success: bool = True
    generated_caption: str
    language_model: str
    top_k: int
    candidate_caption_count: int
    candidate_word_count: int
    candidate_captions_sample: list[str] = Field(default_factory=list)
    retrieved_images: list[RetrievedImage]
    processing: ProcessingInfo


class HealthResponse(BaseModel):
    status: str = "ok"


class ConfigResponse(BaseModel):
    default_top_k: int
    default_model: str
    supported_models: list[str]
    allowed_top_k_range: list[int]
    max_upload_mb: int
    allowed_extensions: list[str]
