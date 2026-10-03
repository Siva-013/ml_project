"""Phase 10 tests: FastAPI backend (health, caption generation, validation,
security, cleanup). The service loads the real Phase 4–8 artifacts once per
module; generation requests use a real Flickr8k image."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.captions.vocabulary import END, START

from backend import config as backend_config
from backend import main as backend_main
from backend.main import app

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


# ---------------------------------------------------------------------------
# Health and config
# ---------------------------------------------------------------------------


def test_health(client: TestClient):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_config_endpoint_exposes_no_paths(client: TestClient):
    response = client.get("/api/config")
    assert response.status_code == 200
    payload = response.json()
    assert payload["default_top_k"] == backend_config.DEFAULT_TOP_K
    assert payload["default_model"] == "trigram"
    assert payload["supported_models"] == ["unigram", "bigram", "trigram"]
    assert "D:\\" not in response.text and "/home/" not in response.text


# ---------------------------------------------------------------------------
# Caption endpoint (requires real artifacts)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not PHASE6_AVAILABLE, reason="data/captions not found — run Phase 6 first"
)
class TestCaptionEndpoint:
    def test_valid_upload_returns_caption(self, client: TestClient, real_image_bytes: bytes):
        response = _upload(client, real_image_bytes)
        assert response.status_code == 200
        payload = response.json()
        assert payload["success"] is True
        assert isinstance(payload["generated_caption"], str)
        assert payload["generated_caption"]
        assert payload["language_model"] == "trigram"
        assert payload["top_k"] == 5
        assert 1 <= payload["candidate_caption_count"] <= 25
        assert payload["candidate_word_count"] > 0
        assert payload["processing"]["feature_dimension"] == 1804
        assert payload["processing"]["similarity_metric"] == "cosine"

    def test_response_structure_exact(self, client: TestClient, real_image_bytes: bytes):
        payload = _upload(client, real_image_bytes).json()
        assert set(payload) == {
            "success",
            "generated_caption",
            "language_model",
            "top_k",
            "candidate_caption_count",
            "candidate_word_count",
            "candidate_captions_sample",
            "retrieved_images",
            "processing",
        }
        assert len(payload["retrieved_images"]) == 5
        for retrieved in payload["retrieved_images"]:
            assert set(retrieved) == {"image_id", "similarity"}
            assert -1.0 - 1e-6 <= retrieved["similarity"] <= 1.0 + 1e-6  # float32 rounding
        # the candidate sample never leaks boundary tokens
        for sample in payload["candidate_captions_sample"]:
            assert START not in sample.split() and END not in sample.split()

    def test_retrieved_images_are_gallery_images(
        self, client: TestClient, real_image_bytes: bytes
    ):
        payload = _upload(client, real_image_bytes).json()
        gallery_ids = set(backend_main.SERVICE.gallery_ids)
        assert all(r["image_id"] in gallery_ids for r in payload["retrieved_images"])

    def test_default_top_k(self, client: TestClient, real_image_bytes: bytes):
        assert _upload(client, real_image_bytes).json()["top_k"] == 5

    def test_custom_top_k(self, client: TestClient, real_image_bytes: bytes):
        payload = _upload(client, real_image_bytes, top_k=3).json()
        assert payload["top_k"] == 3
        assert len(payload["retrieved_images"]) == 3

    def test_model_selection(self, client: TestClient, real_image_bytes: bytes):
        payload = _upload(client, real_image_bytes, model="bigram").json()
        assert payload["language_model"] == "bigram"
        assert payload["generated_caption"]

    def test_invalid_model_rejected(self, client: TestClient, real_image_bytes: bytes):
        response = _upload(client, real_image_bytes, model="gpt")
        assert response.status_code == 422

    def test_invalid_top_k_rejected(self, client: TestClient, real_image_bytes: bytes):
        assert _upload(client, real_image_bytes, top_k=0).status_code == 422
        assert _upload(client, real_image_bytes, top_k=999).status_code == 422

    def test_invalid_lengths_rejected(self, client: TestClient, real_image_bytes: bytes):
        assert _upload(client, real_image_bytes, max_length=999).status_code == 422
        assert _upload(client, real_image_bytes, min_length=99).status_code == 422


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
        response = client.get("/api/gallery/does_not_exist.jpg")
        assert response.status_code == 404

    def test_path_traversal_blocked(self, client: TestClient):
        for malicious in ("..%2F..%2Fconfig.yaml", "..\\..\\config.yaml", "....jpg"):
            response = client.get(f"/api/gallery/{malicious}")
            assert response.status_code == 404, malicious
        # a valid-looking traversal that decodes to a real file outside the
        # gallery must still be rejected because the ID is not in the gallery
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
def test_temporary_upload_files_are_cleaned(client: TestClient, real_image_bytes: bytes):
    temp_dir = Path(backend_main.SERVICE._temp_dir)
    assert temp_dir.exists()
    _upload(client, real_image_bytes)
    assert list(temp_dir.iterdir()) == []  # no upload residue after the request
