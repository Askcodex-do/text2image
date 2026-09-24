"""Face-aware editing pipeline.

Implements the conceptual pipeline from the specification:

    Original Image -> Image Validation -> Face Detection -> Face/Identity
    Reference -> Prompt Builder -> Image Generation -> Identity Checking ->
    Final Image

Face detection is optional.  When no face is found the pipeline continues
normally and records a warning rather than failing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ..models import (
    GeneratedImage,
    GenerationRequest,
    IdentityCheckResult,
    PipelineContext,
    ProviderCapabilities,
)
from ..providers.base import ImageProvider, ProviderError
from .face_service import FaceDetector, IdentityVerifier
from .prompt_builder import builder_for
from .storage import Storage

#: Prompt dialect per provider, so no single universal prompt is used.
PROVIDER_DIALECTS = {
    "remote": "openai",
    "cloud": "stable_diffusion",
    "local": "stable_diffusion",
}


@dataclass
class PipelineResult:
    images: list[GeneratedImage]
    context: PipelineContext

    def to_dict(self) -> dict[str, Any]:
        return {
            "images": [img.to_dict() for img in self.images],
            "pipeline": self.context.to_dict(),
        }


class ImagePipeline:
    def __init__(
        self,
        storage: Storage,
        detector: FaceDetector | None = None,
        verifier: IdentityVerifier | None = None,
        run_identity_check: bool = True,
    ) -> None:
        self.storage = storage
        self.detector = detector or FaceDetector()
        self.verifier = verifier or IdentityVerifier(self.detector)
        self.run_identity_check = run_identity_check

    # -- Public API -------------------------------------------------------
    def execute(self, request: GenerationRequest, provider: ImageProvider) -> PipelineResult:
        caps = provider.capabilities()
        self._validate(request, provider, caps)

        context = PipelineContext(original_image=None)
        if request.is_edit:
            context.original_image = self._resolve_original(request.input_image)

        # Face detection (optional, "do not require face detection").
        if context.original_image:
            self._detect_faces(request, context)

        self._build_prompt(request, context, provider, caps)
        self._select_provider_strategy(request, context, caps)

        try:
            if request.is_edit:
                images = provider.edit(request, context)
            else:
                images = provider.generate(request, context)
        except ProviderError as exc:
            raise PipelineError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise PipelineError(f"Image generation failed: {exc}") from exc

        if not images:
            raise PipelineError("The provider returned no images.")

        self._verify_identity(context, images, caps)
        return PipelineResult(images=images, context=context)

    # -- Stages -----------------------------------------------------------
    def _validate(
        self,
        request: GenerationRequest,
        provider: ImageProvider,
        caps: ProviderCapabilities,
    ) -> None:
        if not request.prompt:
            raise PipelineError("A description is required.")
        if request.number_of_images > caps.max_images_per_request:
            request.number_of_images = caps.max_images_per_request
        if request.is_edit and not provider.is_available():
            reason = provider.unavailable_reason() or "The provider is unavailable."
            raise PipelineError(reason)
        if not request.is_edit and request.input_image:
            raise PipelineError("The provided input image could not be found.")

    def _resolve_original(self, input_image: str | None) -> str | None:
        if not input_image:
            raise PipelineError("No input image was provided for this edit.")
        # Accept either a stored original id or a relative path inside storage.
        stored = self.storage.original_path(os.path.basename(input_image).rsplit(".", 1)[0])
        if stored:
            return stored
        resolved = self.storage.resolve_image(input_image)
        if resolved:
            return resolved
        raise PipelineError(
            "The original image could not be found. Please upload it again."
        )

    def _detect_faces(self, request: GenerationRequest, context: PipelineContext) -> None:
        context.faces = self.detector.detect(context.original_image)
        if not context.faces:
            return
        if request.selected_face_indices:
            context.preserved_face_indices = [
                index for index in request.selected_face_indices
                if any(face.index == index for face in context.faces)
            ]
        if not context.preserved_face_indices:
            context.preserved_face_indices = [face.index for face in context.faces]

    def _build_prompt(
        self,
        request: GenerationRequest,
        context: PipelineContext,
        provider: ImageProvider,
        caps: ProviderCapabilities,
    ) -> None:
        builder = builder_for(PROVIDER_DIALECTS.get(provider.name, "generic"))
        builder.build(request, context, caps)

    def _select_provider_strategy(
        self,
        request: GenerationRequest,
        context: PipelineContext,
        caps: ProviderCapabilities,
    ) -> None:
        """Mark whether a genuine identity reference will be used.

        This flag is what distinguishes *real* identity preservation from a
        prompt-only approximation, and drives the honesty of the UI copy.

        The user-facing warning for a provider that cannot preserve identity is
        emitted by the prompt builder, which also knows whether the backend
        interprets the prompt, so no duplicate note is added here.
        """
        context.provider_uses_identity_reference = bool(
            request.identity_requested()
            and caps.supports_face_preservation
            and context.original_image
            and context.faces
        )

    def _verify_identity(
        self,
        context: PipelineContext,
        images: list[GeneratedImage],
        caps: ProviderCapabilities,
    ) -> None:
        if not (self.run_identity_check and context.original_image and context.faces):
            for image in images:
                image.identity_check = IdentityCheckResult(
                    performed=False,
                    verdict="not_performed",
                    message="Identity check skipped.",
                )
            return
        for image in images:
            generated_path = os.path.join(self.storage.root, image.path)
            image.identity_check = self.verifier.compare(
                context.original_image, generated_path
            )

    # -- Face detection helper for the UI ---------------------------------
    def detect_faces(self, image_id: str) -> list[dict[str, Any]]:
        path = self.storage.original_path(image_id)
        if not path:
            raise PipelineError("Unknown image id.")
        return [face.to_dict() for face in self.detector.detect(path)]


class PipelineError(RuntimeError):
    """Raised for user-facing pipeline failures."""