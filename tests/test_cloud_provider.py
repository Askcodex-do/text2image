"""Cloud provider tests.

The cloud provider performs network I/O, so these tests inject a fake
``urlopen`` rather than hitting the live service.  The fake mirrors the real
endpoint's contract: it returns raw image bytes and ignores any reference image,
which is exactly the limitation the provider must report honestly.
"""

from __future__ import annotations

import os
import urllib.parse

import cv2
import numpy as np
import pytest

from ai_image_studio.models import GenerationRequest
from ai_image_studio.providers.cloud_provider import CloudImageProvider
from ai_image_studio.services.pipeline import ImagePipeline
from ai_image_studio.services.storage import Storage


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc) -> bool:
        return False


def _png_bytes(seed: int, size: int = 64) -> bytes:
    """A deterministic image whose content depends on the requested URL."""
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()


@pytest.fixture()
def captured(monkeypatch):
    """Patch urlopen and record every requested URL."""
    calls: list[str] = []

    def fake_urlopen(request, timeout=None):
        url = request.full_url if hasattr(request, "full_url") else str(request)
        calls.append(url)
        prompt = urllib.parse.unquote(url.split("/prompt/", 1)[1].split("?", 1)[0])
        return _FakeResponse(_png_bytes(abs(hash(prompt)) % 10_000))

    monkeypatch.setattr(
        "ai_image_studio.providers.cloud_provider.urllib.request.urlopen", fake_urlopen
    )
    return calls


def _provider(storage: Storage) -> CloudImageProvider:
    return CloudImageProvider({"storage": storage, "endpoint": "https://example.invalid/prompt"})


def test_cloud_provider_honors_prompt_but_not_face_reference():
    caps = CloudImageProvider({"storage": None}).capabilities()
    assert caps.honors_prompt is True
    assert caps.supports_face_preservation is False
    assert caps.supports_reference_image is False
    # No graded identity control is advertised when identity cannot be preserved.
    assert caps.strength_mapping == {}


def test_local_provider_reports_it_does_not_honor_prompt():
    from ai_image_studio.providers.local_provider import LocalStyleProvider

    caps = LocalStyleProvider({"storage": None}).capabilities()
    assert caps.honors_prompt is False
    assert caps.supports_face_preservation is True


def test_generate_creates_image_from_prompt_only(output_dir, captured):
    storage = Storage(output_dir)
    provider = _provider(storage)
    pipeline = ImagePipeline(storage=storage, run_identity_check=True)

    request = GenerationRequest(
        prompt="chinese prince wearing a black robe and a gold crown",
        style="oil_painting_realism",
        number_of_images=1,
    )
    result = pipeline.execute(request, provider)

    assert len(result.images) == 1
    image = cv2.imread(os.path.join(storage.root, result.images[0].path))
    assert image is not None and image.size > 0

    requested = urllib.parse.unquote(captured[0])
    assert "chinese prince wearing a black robe and a gold crown" in requested
    # The style preset is carried in the prompt since there is no reference.
    assert "oil painting" in requested.lower()
    # No reference image is sent: text-only backend.
    assert "reference" not in requested.lower()


def test_different_prompts_produce_different_images(output_dir, captured):
    storage = Storage(output_dir)
    provider = _provider(storage)
    pipeline = ImagePipeline(storage=storage, run_identity_check=False)

    def render(prompt: str) -> np.ndarray:
        request = GenerationRequest(prompt=prompt, style="oil_painting_realism")
        result = pipeline.execute(request, provider)
        return cv2.imread(os.path.join(storage.root, result.images[0].path)).astype(float)

    prince = render("a chinese prince in a black robe and gold crown")
    mountain = render("a snow-covered mountain at sunrise")

    delta = float(np.mean(np.abs(prince - mountain)))
    assert delta > 1.0, "the prompt must actually change the generated image"


def test_edit_composes_subject_clause_without_claiming_reference(output_dir, captured, face_image):
    storage = Storage(output_dir)
    stored = storage.store_upload(face_image, "person.png")
    provider = _provider(storage)
    pipeline = ImagePipeline(storage=storage, run_identity_check=False)

    request = GenerationRequest(
        prompt="Convert this photograph into a realistic oil painting",
        input_image=stored.id,
        style="oil_painting_realism",
        preserve_face=True,
    )
    result = pipeline.execute(request, provider)

    assert result.context.provider_uses_identity_reference is False
    assert any("best-effort" in w for w in result.context.warnings)
    requested = urllib.parse.unquote(captured[0])
    assert "realistic oil painting" in requested.lower()
    assert result.images[0].metadata["identity_preservation"].startswith("prompt-only")


def test_cloud_provider_uses_configurable_endpoint(monkeypatch):
    monkeypatch.delenv("IMAGE_GEN_URL", raising=False)
    # An unset endpoint falls back to the hosted default.
    provider = CloudImageProvider({"storage": None})
    assert provider.is_available() is True
    assert provider.endpoint == "https://image.pollinations.ai/prompt"

    # An explicit endpoint (e.g. a self-hosted service) wins.
    custom = CloudImageProvider({"storage": None, "endpoint": "https://mine.example/api"})
    assert custom.endpoint == "https://mine.example/api"
    assert custom.is_available() is True


def test_cloud_provider_reads_endpoint_from_environment(monkeypatch):
    monkeypatch.setenv("IMAGE_GEN_URL", "https://env.example/gen")
    monkeypatch.setenv("IMAGE_GEN_MODEL", "sdxl")
    provider = CloudImageProvider({"storage": None})
    assert provider.endpoint == "https://env.example/gen"
    assert provider.model == "sdxl"
