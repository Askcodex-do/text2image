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

from ai_image_studio.models import FacePreservationStrength, GenerationRequest
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


def test_cloud_provider_honors_prompt_and_composites_face():
    caps = CloudImageProvider({"storage": None}).capabilities()
    assert caps.honors_prompt is True
    # The hosted endpoint takes no reference image, but the original face is
    # composited onto the output, so real preservation is available.
    assert caps.supports_face_preservation is True
    assert caps.supports_reference_image is False
    assert set(caps.strength_mapping) == {"low", "medium", "high", "maximum"}


def test_cloud_provider_can_disable_face_compositing(monkeypatch):
    monkeypatch.setenv("IMAGE_GEN_FACE_COMPOSITE", "0")
    caps = CloudImageProvider({"storage": None}).capabilities()
    assert caps.supports_face_preservation is False
    assert caps.strength_mapping == {}
    assert any("prompt only" in n or "not guaranteed" in n for n in caps.notes)


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


def test_edit_sends_no_reference_but_marks_identity_active(output_dir, captured, face_image):
    """The hosted endpoint takes no reference image, yet the provider preserves
    identity by compositing -- so the flag is true while no reference is sent."""
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

    assert result.context.identity_preservation_active is True
    # The provider reports truthfully that it takes no reference image.
    assert provider.capabilities().supports_reference_image is False
    assert not any("best-effort" in w for w in result.context.warnings)
    requested = urllib.parse.unquote(captured[0])
    assert "realistic oil painting" in requested.lower()
    # The fake endpoint returns noise with no detectable face, so compositing is
    # honestly recorded as skipped rather than silently claimed.
    assert result.images[0].metadata["identity_preservation"].startswith("skipped")


def test_face_compositing_puts_the_original_face_into_the_output(
    output_dir, face_image, monkeypatch
):
    """When the generator returns a detectable face, the user's own facial
    pixels must appear in the result -- the actual fix for "it does not use the
    face I gave it"."""
    storage = Storage(output_dir)
    stored = storage.store_upload(face_image, "person.png")
    provider = _provider(storage)

    # Stand in for a generator that produced a portrait-like image: reuse the
    # uploaded photo shifted in tone, so a face is detectable but the pixels
    # differ from the original.
    generated = cv2.imread(face_image)
    tinted = cv2.convertScaleAbs(generated, alpha=0.9, beta=40)
    ok, buffer = cv2.imencode(".png", tinted)
    assert ok
    generated_bytes = buffer.tobytes()

    monkeypatch.setattr(
        "ai_image_studio.providers.cloud_provider.urllib.request.urlopen",
        lambda request, timeout=None: _FakeResponse(generated_bytes),
    )

    pipeline = ImagePipeline(storage=storage, run_identity_check=False)
    request = GenerationRequest(
        prompt="a realistic oil painting",
        input_image=stored.id,
        style="oil_painting_realism",
        preserve_face=True,
        face_preservation_strength=FacePreservationStrength.HIGH,
    )
    result = pipeline.execute(request, provider)

    metadata = result.images[0].metadata["identity_preservation"]
    assert metadata.startswith("face-region compositing"), metadata

    # The output must now be closer to the original face than the raw generated
    # (tinted) image was: that is what compositing is for.
    output = cv2.imread(os.path.join(storage.root, result.images[0].path)).astype(float)
    original = generated.astype(float)
    tinted_f = tinted.astype(float)
    original_distance = float(np.mean(np.abs(output - original)))
    raw_distance = float(np.mean(np.abs(tinted_f - original)))
    assert original_distance < raw_distance, (
        "compositing should move the result toward the user's own face"
    )


def test_distinct_images_per_index_in_a_batch(output_dir, captured, monkeypatch):
    """Regression: the endpoint is deterministic, so a 4-image batch used to be
    four byte-identical copies because no seed was sent."""
    storage = Storage(output_dir)
    provider = _provider(storage)

    seen_urls: list[str] = []

    def fake_urlopen(request, timeout=None):
        url = request.full_url
        seen_urls.append(url)
        # Mimic the real service: identical URL -> identical bytes.
        return _FakeResponse(_png_bytes(abs(hash(url)) % 10_000))

    monkeypatch.setattr(
        "ai_image_studio.providers.cloud_provider.urllib.request.urlopen", fake_urlopen
    )

    pipeline = ImagePipeline(storage=storage, run_identity_check=False)
    request = GenerationRequest(
        prompt="a red apple", style="oil_painting_realism", number_of_images=4
    )
    result = pipeline.execute(request, provider)

    assert len(result.images) == 4
    seeds = [img.seed for img in result.images]
    assert all(s is not None for s in seeds), "every image must carry a seed"
    assert len(set(seeds)) == 4, f"seeds must differ per image, got {seeds}"

    hashes = set()
    for img in result.images:
        payload = open(os.path.join(storage.root, img.path), "rb").read()
        hashes.add(payload)
    assert len(hashes) == 4, "each requested image must be distinct"


def test_user_seed_makes_the_batch_reproducible(output_dir, captured):
    storage = Storage(output_dir)
    provider = _provider(storage)
    pipeline = ImagePipeline(storage=storage, run_identity_check=False)

    def seeds_for() -> list[int]:
        request = GenerationRequest(
            prompt="a red apple", style="oil_painting", number_of_images=3, seed=100
        )
        return [img.seed for img in pipeline.execute(request, provider).images]

    assert seeds_for() == [100, 101, 102]
    assert seeds_for() == [100, 101, 102]


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


def _http_error(code: int, body: str):
    import io
    import urllib.error

    return urllib.error.HTTPError(
        "https://example.invalid/prompt/x", code, "err", {}, io.BytesIO(body.encode())
    )


def test_quota_error_is_explained_and_retried(output_dir, monkeypatch):
    """A quota failure must not surface as a raw HTTP dump, and must be retried
    before giving up."""
    storage = Storage(output_dir)
    provider = CloudImageProvider(
        {
            "storage": storage,
            "endpoint": "https://example.invalid/prompt",
            "retries": 2,
            "retry_delay": 0,
        }
    )
    attempts = {"n": 0}

    def fake_urlopen(request, timeout=None):
        attempts["n"] += 1
        raise _http_error(
            500, '{"error":"Gen Sana request failed with 402: INSUFFICIENT_BALANCE"}'
        )

    monkeypatch.setattr(
        "ai_image_studio.providers.cloud_provider.urllib.request.urlopen", fake_urlopen
    )
    monkeypatch.setattr(
        "ai_image_studio.providers.cloud_provider.time.sleep", lambda _s: None
    )

    pipeline = ImagePipeline(storage=storage, run_identity_check=False)
    request = GenerationRequest(prompt="a cat", style="oil_painting")
    with pytest.raises(Exception) as excinfo:
        pipeline.execute(request, provider)

    assert attempts["n"] == 3, "the retry count should be honoured"
    message = str(excinfo.value)
    assert "quota" in message.lower()
    assert "insufficient balance" in message.lower()
    assert "Local provider" in message


def test_client_error_is_not_retried(output_dir, monkeypatch):
    storage = Storage(output_dir)
    provider = CloudImageProvider(
        {
            "storage": storage,
            "endpoint": "https://example.invalid/prompt",
            "retries": 3,
            "retry_delay": 0,
        }
    )
    attempts = {"n": 0}

    def fake_urlopen(request, timeout=None):
        attempts["n"] += 1
        raise _http_error(400, "bad request")

    monkeypatch.setattr(
        "ai_image_studio.providers.cloud_provider.urllib.request.urlopen", fake_urlopen
    )

    pipeline = ImagePipeline(storage=storage, run_identity_check=False)
    with pytest.raises(Exception):
        pipeline.execute(GenerationRequest(prompt="a cat"), provider)
    assert attempts["n"] == 1, "a definitive client error must not be retried"
