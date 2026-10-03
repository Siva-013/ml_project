"""Phase 10 (revision) — OpenAI vision-language captioning service.

An **external, additional** captioning method: the uploaded image is sent to
the OpenAI Responses API (image-capable model) and its factual caption is
returned verbatim to the frontend. This is NOT part of the classical ML
baseline and is NOT trained by this project — it is a hosted third-party
model, clearly labelled as such in the UI.

Secrets: the API key is read from the environment (``OPENAI_API_KEY``,
loaded from ``.env`` by the backend config). It is never logged, never
returned in responses, and never sent to the frontend — the frontend only
learns *whether* the API is configured.

Failure policy: every foreseeable failure is mapped to a short, safe
message (authentication, rate limit, timeout, connection, unexpected
response) so ``/api/caption`` can return the classical caption alongside
``"error": "..."`` instead of failing the whole request.
"""

from __future__ import annotations

import base64

from openai import (
    APIConnectionError,
    APIStatusError,
    OpenAI,
)

#: Instruction sent with the image (concise, factual, no invention).
CAPTION_PROMPT = (
    "Generate one concise, factual caption describing the main visible "
    "content of this image.\n\n"
    "Mention the main subjects, their important action, and the relevant "
    "setting when clearly visible.\n\n"
    "Do not invent objects, people, actions, locations, or relationships "
    "that are not visually supported.\n\n"
    "Return only the caption."
)

_PROVIDER = "OpenAI"

_client: OpenAI | None = None


class VisionAPIError(Exception):
    """A vision-API failure, carrying a short user-safe message."""


def configure(model: str, api_key: str | None, timeout_seconds: float) -> None:
    """Initialise the module-level OpenAI client (called once at startup).

    With an empty/missing key the client is left unconfigured and
    :func:`is_configured` reports ``False`` — the classical pipeline keeps
    working and the UI shows "Vision API: Not configured".
    """
    global _client, _model, _timeout
    _model = model
    _timeout = timeout_seconds
    _client = OpenAI(api_key=api_key, timeout=timeout_seconds) if api_key else None


def is_configured() -> bool:
    """True when an API key was provided at startup."""
    return _client is not None


def model_name() -> str:
    return _model


def provider() -> str:
    return _PROVIDER


def caption_image(image_bytes: bytes, mime_type: str) -> str:
    """Send the actual uploaded image to the Responses API, return its caption.

    The image travels as a base64 data URL inside an ``input_image`` content
    part of the current Responses API.

    Raises:
        VisionAPIError: With a short, user-safe message for authentication
            failures, rate limits, timeouts, connection errors and
            unexpected responses.
    """
    if _client is None:
        raise VisionAPIError("Vision API not configured")
    data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    try:
        response = _client.responses.create(
            model=_model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": CAPTION_PROMPT},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ],
        )
    except APIConnectionError:
        raise VisionAPIError("Vision API network error") from None
    except APIStatusError as error:
        status = getattr(error, "status_code", None)
        if status == 401:
            raise VisionAPIError("Vision API authentication failed") from None
        if status == 429:
            raise VisionAPIError("Vision API rate limit reached") from None
        if status == 408:
            raise VisionAPIError("Vision API timeout") from None
        raise VisionAPIError("Vision API returned an error") from None
    except Exception:
        raise VisionAPIError("Vision API unavailable") from None

    caption = getattr(response, "output_text", None)
    if not caption or not str(caption).strip():
        raise VisionAPIError("Vision API returned an unexpected response")
    return str(caption).strip()
