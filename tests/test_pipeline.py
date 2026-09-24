"""Pipeline and provider behaviour tests."""

from __future__ import annotations

import os

import cv2

from ai_image_studio.models import (
    FacePreservationStrength,
    GenerationRequest,
    ProviderCapabilities,
)
from ai_image_studio.providers.base import ImageProvider
from ai_image_studio.providers.local_provider import LocalStyleProvider
from ai_image_studio.providers.remote_provider import RemoteImageProvider
from ai_image_studio.services.face_service import FaceDetector, IdentityVerifier
from ai_image_studio.services.pipeline import ImagePipeline, PipelineError
from ai_image_studio.services.prompt_builder import builder_for
from ai_image_studio.services.storage import Storage


def pipeline_and_storage(output_dir):
    storage = Storage(output_dir)
    pipeline = ImagePipeline(storage=storage, run_identity_check=True)
    return pipeline, storage


def upload(pipeline, storage, image_path, name="person.png"):
    stored = storage.store_upload(image_path, name)
    return stored


def local_provider(storage):
    return LocalStyleProvider({"storage": storage})


# -- Capability honesty ---------------------------------------------------
def test_local_provider_declares_face_preservation():
    provider = LocalStyleProvider({"storage": None})
    assert provider.supports_face_preservation() is True
    caps = provider.capabilities()
    assert caps.supports_reference_image is True
    assert caps.supports_multiple_faces is True
    # Every level is selectable; the mapping defines what each one does.
    assert caps.supported_strengths() == ["off", "low", "medium", "high", "maximum"]
    assert set(caps.strength_mapping) == {"low", "medium", "high", "maximum"}


def test_remote_provider_without_config_does_not_claim_support(monkeypatch):
    monkeypatch.delenv("IMAGE_API_URL", raising=False)
    monkeypatch.delenv("IMAGE_API_KEY", raising=False)
    provider = RemoteImageProvider({"storage": None})
    assert provider.is_available() is False
    assert provider.supports_face_preservation() is False
    caps = provider.capabilities()
    # No graded identity control is advertised when it cannot be honoured.
    assert caps.strength_mapping == {}


def test_remote_provider_reports_backend_capabilities(monkeypatch):
    provider = RemoteImageProvider(
        {
            "storage": None,
            "api_url": "https://example.invalid",
            "remote_capabilities": {
                "supports_face_preservation": True,
                "supports_multiple_faces": True,
            },
        }
    )
    assert provider.supports_face_preservation() is True
    assert provider.capabilities().supports_multiple_faces is True


def test_base_provider_default_is_false():
    class Bare(ImageProvider):
        name = "bare"

        def capabilities(self):
            return ProviderCapabilities()

        def generate(self, request, context):
            return []

        def edit(self, request, context):
            return []

    assert Bare().supports_face_preservation() is False
    assert Bare().capabilities().strength_mapping == {}


# -- Face detection -------------------------------------------------------
def test_face_detector_finds_synthetic_face(face_image):
    faces = FaceDetector().detect(face_image)
    assert len(faces) >= 1
    assert faces[0].area > 0


def test_face_detector_returns_empty_for_landscape(blank_image):
    assert FaceDetector().detect(blank_image) == []


# -- Storage safety -------------------------------------------------------
def test_original_is_never_modified(face_image, output_dir):
    storage = Storage(output_dir)
    before = open(face_image, "rb").read()
    stored = storage.store_upload(face_image, "person.png")
    pipeline = ImagePipeline(storage=storage)
    request = GenerationRequest(
        prompt="oil painting",
        input_image=stored.id,
        style="oil_painting_realism",
        preserve_face=True,
    )
    result = pipeline.execute(request, local_provider(storage))
    assert len(result.images) == 1
    after = open(face_image, "rb").read()
    assert before == after, "the user's original file must remain untouched"
    assert os.path.exists(stored.path)
    original_before = open(stored.path, "rb").read()
    # Regenerating must not change the stored original either.
    pipeline.execute(request, local_provider(storage))
    assert open(stored.path, "rb").read() == original_before


def test_generated_name_is_human_readable(face_image, output_dir):
    storage = Storage(output_dir)
    stored = storage.store_upload(face_image, "person.jpg")
    pipeline = ImagePipeline(storage=storage)
    request = GenerationRequest(
        prompt="watercolor", input_image=stored.id, style="watercolor"
    )
    result = pipeline.execute(request, local_provider(storage))
    name = os.path.basename(result.images[0].path)
    assert name.startswith("person_watercolor_local_")
    assert name.endswith(".png")


def test_path_traversal_is_blocked(output_dir):
    storage = Storage(output_dir)
    assert storage.resolve_image("../../etc/passwd") is None
    assert storage.original_path("../../etc/passwd") is None


# -- Pipeline behaviour ---------------------------------------------------
def test_no_face_continues_normally(blank_image, output_dir):
    storage = Storage(output_dir)
    stored = storage.store_upload(blank_image, "landscape.png")
    pipeline = ImagePipeline(storage=storage)
    request = GenerationRequest(
        prompt="make it a painting",
        input_image=stored.id,
        style="oil_painting",
        preserve_face=True,
    )
    result = pipeline.execute(request, local_provider(storage))
    assert len(result.images) == 1
    assert result.context.faces == []
    assert any("No face was detected" in w for w in result.context.warnings)
    assert result.context.provider_uses_identity_reference is False


def test_identity_reference_is_used_when_face_present(face_image, output_dir):
    storage = Storage(output_dir)
    stored = storage.store_upload(face_image, "person.png")
    pipeline = ImagePipeline(storage=storage)
    request = GenerationRequest(
        prompt="realistic oil painting",
        input_image=stored.id,
        style="oil_painting_realism",
        preserve_face=True,
        face_preservation_strength=FacePreservationStrength.HIGH,
    )
    result = pipeline.execute(request, local_provider(storage))
    assert result.context.faces
    assert result.context.provider_uses_identity_reference is True
    assert result.context.identity_prompt
    metadata = result.images[0].metadata
    assert metadata["face_preservation"] == "face-region compositing"


def test_preserve_face_off_skips_identity(face_image, output_dir):
    storage = Storage(output_dir)
    stored = storage.store_upload(face_image, "person.png")
    pipeline = ImagePipeline(storage=storage)
    request = GenerationRequest(
        prompt="anime",
        input_image=stored.id,
        style="anime",
        preserve_face=False,
        face_preservation_strength=FacePreservationStrength.OFF,
    )
    result = pipeline.execute(request, local_provider(storage))
    assert result.context.identity_prompt == ""
    assert result.context.provider_uses_identity_reference is False


def test_strength_changes_provider_params(output_dir):
    storage = Storage(output_dir)
    provider = local_provider(storage)
    caps = provider.capabilities()
    high = caps.strength_params(FacePreservationStrength.HIGH)
    maximum = caps.strength_params(FacePreservationStrength.MAXIMUM)
    assert high["face_blend"] < maximum["face_blend"]
    assert high["style_strength"] > maximum["style_strength"]


def test_prompt_missing_raises(face_image, output_dir):
    storage = Storage(output_dir)
    stored = storage.store_upload(face_image, "person.png")
    pipeline = ImagePipeline(storage=storage)
    with_dummy = GenerationRequest(prompt="", input_image=stored.id, style="oil_painting")
    try:
        pipeline.execute(with_dummy, local_provider(storage))
        assert False, "expected PipelineError"
    except PipelineError as exc:
        assert "description is required" in str(exc).lower()


def test_identity_check_is_non_biometric(face_image, output_dir):
    storage = Storage(output_dir)
    stored = storage.store_upload(face_image, "person.png")
    pipeline = ImagePipeline(storage=storage)
    request = GenerationRequest(
        prompt="oil painting", input_image=stored.id, style="oil_painting_realism"
    )
    result = pipeline.execute(request, local_provider(storage))
    check = result.images[0].identity_check
    assert check is not None
    assert check.performed is True
    assert check.biometric is False
    assert 0.0 <= check.score <= 1.0


def test_identity_check_never_crashes_without_face(blank_image, output_dir):
    storage = Storage(output_dir)
    stored = storage.store_upload(blank_image, "landscape.png")
    pipeline = ImagePipeline(storage=storage)
    request = GenerationRequest(
        prompt="oil painting", input_image=stored.id, style="oil_painting",
        preserve_face=False, face_preservation_strength=FacePreservationStrength.OFF,
    )
    result = pipeline.execute(request, local_provider(storage))
    check = result.images[0].identity_check
    assert check.performed is False


def test_multiple_images_requested(face_image, output_dir):
    storage = Storage(output_dir)
    stored = storage.store_upload(face_image, "person.png")
    pipeline = ImagePipeline(storage=storage)
    provider = local_provider(storage)
    request = GenerationRequest(
        prompt="cartoon", input_image=stored.id, style="cartoon", number_of_images=3
    )
    result = pipeline.execute(request, provider)
    assert len(result.images) == 3
    paths = [image.path for image in result.images]
    assert len(set(paths)) == 3, "each output must have a unique filename"
    for path in paths:
        name = os.path.basename(path)
        assert name.startswith("person_cartoon_local_")
        assert os.path.exists(os.path.join(storage.root, path))
    # A second batch continues numbering rather than overwriting.
    result2 = pipeline.execute(request, provider)
    assert not set(paths) & {image.path for image in result2.images}


def test_requested_images_capped_by_provider(face_image, output_dir):
    storage = Storage(output_dir)
    stored = storage.store_upload(face_image, "person.png")
    pipeline = ImagePipeline(storage=storage)
    request = GenerationRequest(
        prompt="oil painting",
        input_image=stored.id,
        style="oil_painting",
        number_of_images=99,
    )
    result = pipeline.execute(request, local_provider(storage))
    assert len(result.images) == 8  # local provider's max_images_per_request


def test_all_styles_render_without_error(face_image, output_dir):
    from ai_image_studio.styles import STYLES

    storage = Storage(output_dir)
    stored = storage.store_upload(face_image, "person.png")
    pipeline = ImagePipeline(storage=storage)
    provider = local_provider(storage)
    for style in STYLES:
        if style == "none":
            continue
        request = GenerationRequest(
            prompt="test", input_image=stored.id, style=style, preserve_face=True
        )
        result = pipeline.execute(request, provider)
        assert result.images, f"style {style} produced no image"
        image = cv2.imread(os.path.join(storage.root, result.images[0].path))
        assert image is not None and image.size > 0


# -- Prompt builder -------------------------------------------------------
def test_prompt_builder_is_provider_specific(face_image, output_dir):
    storage = Storage(output_dir)
    stored = storage.store_upload(face_image, "person.png")
    request = GenerationRequest(
        prompt="convert into an oil painting",
        input_image=stored.id,
        style="oil_painting_realism",
        preserve_face=True,
        preserve_composition=True,
    )
    faces = FaceDetector().detect(stored.path)

    from ai_image_studio.models import PipelineContext

    sd_context = PipelineContext(original_image=stored.path, faces=faces,
                                 preserved_face_indices=[f.index for f in faces])
    openai_context = PipelineContext(original_image=stored.path, faces=faces,
                                     preserved_face_indices=[f.index for f in faces])
    caps = LocalStyleProvider({"storage": storage}).capabilities()
    builder_for("stable_diffusion").build(request, sd_context, caps)
    builder_for("openai").build(request, openai_context, caps)

    assert sd_context.effective_prompt != openai_context.effective_prompt
    assert "," in sd_context.effective_prompt
    assert sd_context.identity_prompt
    assert sd_context.composition_prompt


def test_explicit_expression_override_wins(face_image, output_dir):
    storage = Storage(output_dir)
    stored = storage.store_upload(face_image, "person.png")
    faces = FaceDetector().detect(stored.path)
    from ai_image_studio.models import PipelineContext

    request = GenerationRequest(
        prompt="make the person smile naturally",
        input_image=stored.id,
        style="oil_painting_realism",
        preserve_face=True,
        preserve_expression=True,
    )
    context = PipelineContext(original_image=stored.path, faces=faces,
                              preserved_face_indices=[f.index for f in faces])
    caps = LocalStyleProvider({"storage": storage}).capabilities()
    builder_for("generic").build(request, context, caps)
    assert context.expression_prompt == ""
    assert any("explicitly changes the expression" in w for w in context.warnings)


def test_unsupported_provider_adds_honest_warning(face_image, output_dir):
    storage = Storage(output_dir)
    stored = storage.store_upload(face_image, "person.png")
    faces = FaceDetector().detect(stored.path)
    from ai_image_studio.models import PipelineContext

    request = GenerationRequest(
        prompt="oil painting", input_image=stored.id, style="oil_painting_realism",
        preserve_face=True,
    )
    context = PipelineContext(original_image=stored.path, faces=faces,
                              preserved_face_indices=[f.index for f in faces])
    caps = ProviderCapabilities(supports_face_preservation=False)
    builder_for("generic").build(request, context, caps)
    assert context.identity_prompt == ""
    # A provider that neither uses a reference nor reads the prompt cannot
    # preserve identity at all, and must say so.
    assert any("cannot preserve identity" in w for w in context.warnings)


def test_unsupported_provider_that_honors_prompt_describes_best_effort(face_image, output_dir):
    """A text-only backend is not claimed to be useless -- it is described as
    best-effort, since the subject is carried in the prompt."""
    storage = Storage(output_dir)
    stored = storage.store_upload(face_image, "person.png")
    faces = FaceDetector().detect(stored.path)
    from ai_image_studio.models import PipelineContext

    request = GenerationRequest(
        prompt="oil painting", input_image=stored.id, style="oil_painting_realism",
        preserve_face=True,
    )
    context = PipelineContext(original_image=stored.path, faces=faces,
                              preserved_face_indices=[f.index for f in faces])
    caps = ProviderCapabilities(supports_face_preservation=False, honors_prompt=True)
    builder_for("stable_diffusion").build(request, context, caps)
    assert any("best-effort" in w for w in context.warnings)
    assert not any("has no effect" in w for w in context.warnings)


# -- Identity verifier ----------------------------------------------------
def test_identity_verifier_scores_identical_images_high(face_image):
    verifier = IdentityVerifier()
    same = verifier.compare(face_image, face_image)
    assert same.performed is True
    assert same.score > 0.95
    assert same.verdict == "completed"
    assert same.biometric is False


def test_identity_verifier_flags_replaced_face(face_image, blank_image, asset_dir):
    verifier = IdentityVerifier()
    # A faceless image: no face found in the generated image.
    replaced = verifier.compare(face_image, blank_image)
    assert replaced.performed is True
    assert replaced.verdict == "review_recommended"
    assert replaced.biometric is False

    # A genuinely different person scores far lower than the same person.
    other_face = os.path.join(asset_dir, "test_face2.jpg")
    different = verifier.compare(face_image, other_face)
    same = verifier.compare(face_image, face_image)
    assert different.performed is True
    assert different.score < same.score - 0.4
    assert different.verdict == "review_recommended"


def test_identity_check_tracks_preservation_strength(face_image, output_dir):
    """A stronger preservation setting must yield a higher identity score."""
    storage = Storage(output_dir)
    stored = storage.store_upload(face_image, "person.png")
    pipeline = ImagePipeline(storage=storage)
    provider = local_provider(storage)

    scores = {}
    for strength in (FacePreservationStrength.OFF, FacePreservationStrength.MAXIMUM):
        request = GenerationRequest(
            prompt="oil painting",
            input_image=stored.id,
            style="oil_painting_realism",
            preserve_face=True,
            face_preservation_strength=strength,
        )
        result = pipeline.execute(request, provider)
        scores[strength] = result.images[0].identity_check.score
    assert scores[FacePreservationStrength.MAXIMUM] > scores[FacePreservationStrength.OFF]


def test_face_pixels_are_preserved_at_maximum(face_image, output_dir):
    """At maximum strength the face region stays close to the original."""
    import numpy as np

    storage = Storage(output_dir)
    stored = storage.store_upload(face_image, "person.png")
    faces = FaceDetector().detect(stored.path)
    assert faces
    face = faces[0]
    pipeline = ImagePipeline(storage=storage)
    provider = local_provider(storage)

    original = cv2.imread(stored.path).astype(np.float32)

    def face_delta(strength):
        request = GenerationRequest(
            prompt="oil painting",
            input_image=stored.id,
            style="oil_painting_realism",
            preserve_face=True,
            face_preservation_strength=strength,
        )
        result = pipeline.execute(request, provider)
        generated = cv2.imread(os.path.join(storage.root, result.images[0].path)).astype(np.float32)
        region = generated[face.y:face.y + face.h, face.x:face.x + face.w]
        reference = original[face.y:face.y + face.h, face.x:face.x + face.w]
        return float(np.mean(np.abs(region - reference)))

    assert face_delta(FacePreservationStrength.MAXIMUM) < face_delta(
        FacePreservationStrength.OFF
    )


def test_multi_person_identities_preserved(two_face_image, output_dir):
    """All detected faces are preserved by default in a multi-person image."""
    storage = Storage(output_dir)
    stored = storage.store_upload(two_face_image, "people.png")
    faces = FaceDetector().detect(stored.path)
    assert len(faces) >= 2, "fixture must contain two detectable faces"

    pipeline = ImagePipeline(storage=storage)
    request = GenerationRequest(
        prompt="oil painting",
        input_image=stored.id,
        style="oil_painting_realism",
        preserve_face=True,
        face_preservation_strength=FacePreservationStrength.MAXIMUM,
    )
    result = pipeline.execute(request, local_provider(storage))
    assert result.context.preserved_face_indices == [f.index for f in faces]


def test_style_is_visible_on_the_face_not_only_the_background(face_image, output_dir):
    """Regression: the face used to be copied back almost unchanged, so a
    portrait looked identical to the original photo after 'styling'.

    The face must receive the requested style -- and the style must differ from
    the background-only effect, i.e. the whole image is transformed.
    """
    import numpy as np

    storage = Storage(output_dir)
    stored = storage.store_upload(face_image, "person.png")
    faces = FaceDetector().detect(stored.path)
    assert faces
    face = faces[0]
    pipeline = ImagePipeline(storage=storage)
    original = cv2.imread(stored.path).astype(np.float32)

    request = GenerationRequest(
        prompt="oil painting",
        input_image=stored.id,
        style="oil_painting",
        preserve_face=True,
        face_preservation_strength=FacePreservationStrength.HIGH,
    )
    result = pipeline.execute(request, local_provider(storage))
    generated = cv2.imread(os.path.join(storage.root, result.images[0].path)).astype(np.float32)

    region = generated[face.y:face.y + face.h, face.x:face.x + face.w]
    reference = original[face.y:face.y + face.h, face.x:face.x + face.w]
    face_delta = float(np.mean(np.abs(region - reference)))

    # A meaningful painterly change in the face, not a near-copy.
    assert face_delta > 8.0, (
        f"face changed by only {face_delta:.2f}; the style is not being applied "
        "to the face"
    )


def test_styles_are_visually_distinct(face_image, output_dir):
    """Regression: renaissance/cinematic/concept_art all routed to the same
    filter and rendered pixel-identical output."""
    import numpy as np

    storage = Storage(output_dir)
    stored = storage.store_upload(face_image, "person.png")
    pipeline = ImagePipeline(storage=storage, run_identity_check=False)
    provider = local_provider(storage)

    rendered = {}
    for style in ("oil_painting", "oil_painting_realism", "renaissance",
                  "cinematic", "concept_art", "watercolor"):
        request = GenerationRequest(
            prompt="paint it", input_image=stored.id, style=style,
            preserve_face=False,
        )
        result = pipeline.execute(request, provider)
        rendered[style] = cv2.imread(
            os.path.join(storage.root, result.images[0].path)
        ).astype(np.float32)

    keys = list(rendered)
    for i, first in enumerate(keys):
        for second in keys[i + 1:]:
            delta = float(np.mean(np.abs(rendered[first] - rendered[second])))
            assert delta > 3.0, (
                f"styles {first} and {second} render near-identical output "
                f"(mean abs difference {delta:.2f})"
            )