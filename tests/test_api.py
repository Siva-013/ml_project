"""Phase 10 tests: FastAPI backend (health, dual-caption generation, upload
validation, gallery security, cleanup) and the OpenAI vision-API integration.

The vision API is **always mocked** in automated tests (monkeypatching
``backend.services.vision_api``) — no test ever makes a real paid API call.
The default test environment has no API key, so the unconfigured path is
exercised deterministically as well."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.captions.vocabulary import END, START

from backend import config as backend_config
from backend import main as backend_main
from backend.main import app
from backend.services import vision_api

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAPTIONS_DIR = PROJECT_ROOT / "data" / "captions"
PHASE6_AVAILABLE = (CAPTIONS_DIR / "train_captions.csv").exists()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def real_image_bytes() -> bytes:
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from src.config import load_config, resolve_path

    config = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    image_path = resolve_path(config["paths"]["images_dir"]) / "1000268201_693b08cb0e.jpg"
    return image_path.read_bytes()


def _upload(client: TestClient, real_image_bytes: bytes, **params):
    files = {"image": ("query.jpg", real_image_bytes, "image/jpeg")}
    data = {key: str(value) for key, value in params.items()}
    return client.post("/api/caption", files=files, data=data)


def _configure_vision(monkeypatch: pytest.MonkeyPatch, caption: str | Exception):
    """Force the vision API 'configured' with a stubbed call (no real API)."""
    monkeypatch.setattr(vision_api, "is_configured", lambda: True)

    def fake_caption(image_bytes: bytes, mime_type: str) -> str:
        if isinstance(caption, Exception):
            raise caption
        return caption

    monkeypatch.setattr(vision_api, "caption_image", fake_caption)


def json_dumps(payload) -> str:
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# Health and config
# ---------------------------------------------------------------------------


def test_health(client: TestClient):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_config_endpoint(client: TestClient):
    response = client.get("/api/config")
    assert response.status_code == 200
    payload = response.json()
    assert payload["default_top_k"] == backend_config.DEFAULT_TOP_K
    assert payload["default_model"] == "trigram"
    assert payload["supported_models"] == ["unigram", "bigram", "trigram"]
    # vision status is safe: configured flag + model name, never the key
    assert payload["vision_api"]["provider"] == "OpenAI"
    assert isinstance(payload["vision_api"]["configured"], bool)
    text = response.text
    assert "OPENAI_API_KEY" not in text
    assert "sk-" not in text


# ---------------------------------------------------------------------------
# Caption endpoint: classical block (requires real artifacts)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not PHASE6_AVAILABLE, reason="data/captions not found — run Phase 6 first"
)
class TestClassicalCaption:
    def test_valid_upload_returns_both_blocks(
        self, client: TestClient, real_image_bytes: bytes, monkeypatch: pytest.MonkeyPatch
    ):
        _configure_vision(monkeypatch, "A dog running through a grassy field.")
        response = _upload(client, real_image_bytes)
        assert response.status_code == 200
        payload = response.json()

        assert payload["success"] is True
        assert set(payload) == {"success", "classical", "vision_api"}

        classical = payload["classical"]
        assert classical["caption"]
        assert classical["model"] == "trigram"
        assert classical["model_label"] == "Trigram N-gram"
        assert classical["top_k"] == 5
        assert 1 <= classical["candidate_caption_count"] <= 25
        assert classical["candidate_word_count"] > 0
        assert len(classical["retrieved_images"]) == 5
        assert classical["processing"]["feature_dimension"] == 1804

        vision = payload["vision_api"]
        assert vision["caption"] == "A dog running through a grassy field."
        assert vision["provider"] == "OpenAI"
        assert vision["error"] is None

    def test_classical_caption_survives_vision_failure(
        self, client: TestClient, real_image_bytes: bytes, monkeypatch: pytest.MonkeyPatch
    ):
        _configure_vision(monkeypatch, vision_api.VisionAPIError("Vision API rate limit reached"))
        response = _upload(client, real_image_bytes)
        assert response.status_code == 200
        payload = response.json()
        assert payload["success"] is True
        assert payload["classical"]["caption"]  # classical result intact
        vision = payload["vision_api"]
        assert vision["caption"] is None
        assert vision["error"] == "Vision API rate limit reached"

    def test_vision_authentication_failure_is_safe(
        self, client: TestClient, real_image_bytes: bytes, monkeypatch: pytest.MonkeyPatch
    ):
        _configure_vision(
            monkeypatch, vision_api.VisionAPIError("Vision API authentication failed")
        )
        payload = _upload(client, real_image_bytes).json()
        assert payload["vision_api"]["error"] == "Vision API authentication failed"
        # no secret material is ever echoed
        assert "sk-" not in json_dumps(payload)

    def test_unconfigured_vision_reports_status_without_caption(
        self, client: TestClient, real_image_bytes: bytes, monkeypatch: pytest.MonkeyPatch
    ):
        # deterministic even if the user later sets a real key in .env
        monkeypatch.setattr(vision_api, "is_configured", lambda: False)
        payload = _upload(client, real_image_bytes).json()
        assert payload["classical"]["caption"]
        vision = payload["vision_api"]
        assert vision["caption"] is None
        assert vision["error"] == "Vision API not configured"
        assert vision["provider"] == "OpenAI"

    def test_candidate_sample_never_leaks_boundary_tokens(
        self, client: TestClient, real_image_bytes: bytes, monkeypatch: pytest.MonkeyPatch
    ):
        _configure_vision(monkeypatch, "test caption")
        classical = _upload(client, real_image_bytes).json()["classical"]
        for sample in classical["candidate_captions_sample"]:
            assert START not in sample.split() and END not in sample.split()

    def test_retrieved_images_are_gallery_images(
        self, client: TestClient, real_image_bytes: bytes, monkeypatch: pytest.MonkeyPatch
    ):
        _configure_vision(monkeypatch, "test caption")
        retrieved = _upload(client, real_image_bytes).json()["classical"]["retrieved_images"]
        gallery_ids = set(backend_main.SERVICE.gallery_ids)
        assert all(r["image_id"] in gallery_ids for r in retrieved)

    def test_default_top_k(
        self, client: TestClient, real_image_bytes: bytes, monkeypatch: pytest.MonkeyPatch
    ):
        _configure_vision(monkeypatch, "test caption")
        assert _upload(client, real_image_bytes).json()["classical"]["top_k"] == 5

    def test_custom_top_k(
        self, client: TestClient, real_image_bytes: bytes, monkeypatch: pytest.MonkeyPatch
    ):
        _configure_vision(monkeypatch, "test caption")
        classical = _upload(client, real_image_bytes, top_k=3).json()["classical"]
        assert classical["top_k"] == 3
        assert len(classical["retrieved_images"]) == 3

    def test_model_selection(
        self, client: TestClient, real_image_bytes: bytes, monkeypatch: pytest.MonkeyPatch
    ):
        _configure_vision(monkeypatch, "test caption")
        classical = _upload(client, real_image_bytes, model="bigram").json()["classical"]
        assert classical["model"] == "bigram"
        assert classical["model_label"] == "Bigram N-gram"
        assert classical["caption"]

    def test_invalid_model_rejected(self, client: TestClient, real_image_bytes: bytes):
        assert _upload(client, real_image_bytes, model="gpt").status_code == 422

    def test_invalid_top_k_rejected(self, client: TestClient, real_image_bytes: bytes):
        assert _upload(client, real_image_bytes, top_k=0).status_code == 422
        assert _upload(client, real_image_bytes, top_k=999).status_code == 422

    def test_invalid_lengths_rejected(self, client: TestClient, real_image_bytes: bytes):
        assert _upload(client, real_image_bytes, max_length=999).status_code == 422
        assert _upload(client, real_image_bytes, min_length=99).status_code == 422


def json_dumps_helper(payload) -> str:
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# Upload validation and security
# ---------------------------------------------------------------------------


class TestUploadValidation:
    def test_text_file_rejected_as_invalid_image(self, client: TestClient):
        response = client.post(
            "/api/caption",
            files={"image": ("query.jpg", b"this is not an image", "image/jpeg")},
        )
        assert response.status_code == 400
        assert "not a valid image" in response.json()["detail"]

    def test_unsupported_extension_rejected(self, client: TestClient):
        response = client.post(
            "/api/caption",
            files={"image": ("query.gif", b"GIF89a", "image/gif")},
        )
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_unsupported_content_type_rejected(self, client: TestClient):
        response = client.post(
            "/api/caption",
            files={"image": ("query.jpg", b"\xff\xd8\xff", "text/plain")},
        )
        assert response.status_code == 400

    def test_empty_file_rejected(self, client: TestClient):
        response = client.post(
            "/api/caption",
            files={"image": ("query.jpg", b"", "image/jpeg")},
        )
        assert response.status_code == 400

    def test_oversized_file_rejected_with_413(self, client: TestClient):
        big = b"\xff\xd8\xff" + b"\x00" * (backend_config.MAX_UPLOAD_MB * 1024 * 1024)
        response = client.post(
            "/api/caption",
            files={"image": ("big.jpg", big, "image/jpeg")},
        )
        assert response.status_code == 413

    def test_missing_image_field_rejected(self, client: TestClient):
        response = client.post("/api/caption", data={"top_k": "5"})
        assert response.status_code == 422


class TestGallerySecurity:
    def test_unknown_gallery_image_404(self, client: TestClient):
        assert client.get("/api/gallery/does_not_exist.jpg").status_code == 404

    def test_path_traversal_blocked(self, client: TestClient):
        for malicious in ("..%2F..%2Fconfig.yaml", "..\\..\\config.yaml", "....jpg"):
            response = client.get(f"/api/gallery/{malicious}")
            assert response.status_code == 404, malicious
        response = client.get("/api/gallery/%2E%2E%2FREADME%2Emd")
        assert response.status_code == 404

    def test_gallery_image_served(self, client: TestClient):
        image_id = sorted(backend_main.SERVICE.gallery_ids)[0]
        response = client.get(f"/api/gallery/{image_id}")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("image/")


# ---------------------------------------------------------------------------
# Temporary file handling
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not PHASE6_AVAILABLE, reason="data/captions not found — run Phase 6 first"
)
def test_temporary_upload_files_are_cleaned(
    client: TestClient, real_image_bytes: bytes, monkeypatch: pytest.MonkeyPatch
):
    _configure_vision(monkeypatch, "test caption")
    temp_dir = Path(backend_main.SERVICE._temp_dir)
    assert temp_dir.exists()
    _upload(client, real_image_bytes)
    assert list(temp_dir.iterdir()) == []  # no upload residue after the request


# ---------------------------------------------------------------------------
# .env hygiene
# ---------------------------------------------------------------------------


def test_env_file_is_gitignored():
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore
