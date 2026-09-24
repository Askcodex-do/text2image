"""Cloud image provider backed by a hosted text-to-image service.

This is the backend that actually *creates* what the user describes.  Unlike the
local OpenCV provider it interprets the prompt text, so requests such as
"chinese prince wearing a black robe and a gold crown" produce new content
rather than a filter over the original photo.

Trade-off, stated honestly in :meth:`capabilities`: the hosted model does not
accept a reference image, so it cannot preserve a specific person's face.  When
the user supplies a photo the provider therefore *describes* that photo's
subject in the prompt and reports ``supports_face_preservation=False``, so the
GUI tells the user that identity preservation is best-effort only instead of
pretending otherwise.

The endpoint and model are configurable (``IMAGE_GEN_URL`` / ``IMAGE_GEN_MODEL``)
so a different or self-hosted service can be dropped in without code changes.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from ..models import (
    GeneratedImage,
    GenerationRequest,
    PipelineContext,
    ProviderCapabilities,
)
from .base import ImageProvider, ProviderError
from ..services.face_service import FaceDetector

#: Default hosted endpoint.  Kept overridable so deployments can point at their
#: own service.
DEFAULT_ENDPOINT = "https://image.pollinations.ai/prompt"

_ASPECT_SIZES = {
    "1:1": (768, 768),
    "4:3": (896, 672),
    "3:4": (672, 896),
    "3:2": (960, 640),
    "2:3": (640, 960),
    "16:9": (1024, 576),
    "9:16": (576, 1024),
}

#: Styles that are inherently non-photographic, so a portrait photo cannot be
#: carried into the result.  Used to temper identity claims in the prompt.
_STYLISED = {"anime", "cartoon", "pencil_sketch", "charcoal"}


class CloudImageProvider(ImageProvider):
    name = "cloud"
    label = "Cloud (AI generation)"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.endpoint = (
            self.config.get("endpoint") or os.environ.get("IMAGE_GEN_URL") or DEFAULT_ENDPOINT
        ).strip()
        self.model = (
            self.config.get("model") or os.environ.get("IMAGE_GEN_MODEL") or "flux"
        ).strip()
        self.timeout = int(
            self.config.get("timeout") or os.environ.get("IMAGE_GEN_TIMEOUT") or 120
        )
        self._detector = FaceDetector()

    # -- Capabilities -----------------------------------------------------
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            # No reference-image conditioning on the hosted endpoint, so a
            # specific person's identity cannot be guaranteed.  Reported
            # honestly rather than advertised as supported.
            supports_face_preservation=False,
            supports_reference_image=False,
            supports_multiple_faces=False,
            supports_composition_control=True,
            supports_expression_control=True,
            supports_identity_check=True,
            supports_negative_prompt=False,
            supports_seed=True,
            honors_prompt=True,
            is_remote=True,
            max_images_per_request=4,
            strength_mapping={},
            notes=[
                "Generates a brand-new image from your description, so it can "
                "create people, clothing and scenes that are not in the photo.",
                "This backend does not accept a reference image, so the person's "
                "exact face is not guaranteed; the subject is re-described in "
                "the prompt instead. Results are best-effort.",
            ],
        )

    def is_available(self) -> bool:
        return bool(self.endpoint)

    def unavailable_reason(self) -> str:
        if not self.endpoint:
            return "No cloud generation endpoint is configured."
        return ""

    # -- Generation -------------------------------------------------------
    def generate(self, request: GenerationRequest, context: PipelineContext) -> list[GeneratedImage]:
        return self._call(request, context, edit=False)

    def edit(self, request: GenerationRequest, context: PipelineContext) -> list[GeneratedImage]:
        return self._call(request, context, edit=True)

    def _call(
        self,
        request: GenerationRequest,
        context: PipelineContext,
        edit: bool,
    ) -> list[GeneratedImage]:
        if not self.is_available():
            raise ProviderError(self.unavailable_reason())

        prompt = self._compose_prompt(request, context, edit)
        width, height = _ASPECT_SIZES.get(request.aspect_ratio, (768, 768))
        count = max(1, min(request.number_of_images, 4))

        results: list[GeneratedImage] = []
        for index in range(count):
            seed = request.seed
            if seed is not None:
                seed = int(seed) + index
            image_bytes = self._request_image(prompt, width, height, seed)
            results.append(self._store(image_bytes, request, context, index))
        return results

    # -- Prompt -----------------------------------------------------------
    def _compose_prompt(
        self,
        request: GenerationRequest,
        context: PipelineContext,
        edit: bool,
    ) -> str:
        """Start from the pipeline's provider-specific prompt, then add the
        subject clause this text-only backend needs.

        The prompt builder already keeps user text, style and identity as
        separate clauses; reusing ``context.effective_prompt`` avoids drifting
        from that structure.
        """
        base = context.effective_prompt or request.prompt.strip()
        parts: list[str] = [base]

        if edit and context.original_image and request.identity_requested():
            # No reference image is accepted, so carry the subject across in
            # words.  Be explicit that a close likeness is not guaranteed.
            subject = self._describe_subject(context)
            preserve = request.style not in _STYLISED
            if subject:
                parts.append(("keep the same person: " if preserve else "") + subject)
            if preserve:
                parts.append("recognisable facial features")
        elif edit and context.original_image:
            subject = self._describe_subject(context)
            if subject:
                parts.append(subject)

        if context.composition_prompt:
            parts.append(context.composition_prompt)
        if context.expression_prompt:
            parts.append(context.expression_prompt)

        return ", ".join(p.strip().rstrip(".") for p in parts if p and p.strip())

    def _describe_subject(self, context: PipelineContext) -> str:
        """A crude, honest description of the photographed subject.

        A real face-embedding model would be needed for genuinely faithful
        identity transfer; this just gives the generator enough context to keep
        the framing and subject count, which is all a text-only backend can use.
        """
        faces = context.faces
        if not faces:
            return "a person matching the reference photograph"
        if len(faces) == 1:
            return "a portrait of one person"
        return f"a group portrait of {len(faces)} people"

    # -- HTTP -------------------------------------------------------------
    def _request_image(
        self, prompt: str, width: int, height: int, seed: int | None
    ) -> bytes:
        params: dict[str, Any] = {
            "width": width,
            "height": height,
            "nologo": "true",
            "model": self.model,
        }
        if seed is not None:
            params["seed"] = seed
        url = f"{self.endpoint.rstrip('/')}/{urllib.parse.quote(prompt)}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise ProviderError(
                f"The generation service returned HTTP {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderError(f"Could not reach the generation service: {exc}") from exc
        if not payload:
            raise ProviderError("The generation service returned an empty response.")
        return payload

    def _store(
        self,
        payload: bytes,
        request: GenerationRequest,
        context: PipelineContext,
        index: int,
    ) -> GeneratedImage:
        import cv2
        import numpy as np

        from ..services.storage import Storage

        storage: Storage = self.config["storage"]
        array = np.frombuffer(payload, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise ProviderError(
                "The generation service returned data that is not a valid image."
            )
        destination = storage.generated_path(
            original_filename=request.input_image or "image",
            style_key=request.style,
            provider_name=self.name,
            index=index,
        )
        cv2.imwrite(destination, image)
        return GeneratedImage(
            path=storage.relative_image_path(destination),
            index=index,
            original_path=(
                storage.relative_image_path(context.original_image)
                if context.original_image
                else None
            ),
            seed=request.seed,
            provider=self.name,
            metadata={
                "engine": "cloud",
                "model": self.model,
                "prompt": request.prompt,
                "identity_preservation": "prompt-only (no reference image support)",
            },
        )

    def describe(self) -> dict[str, Any]:
        data = super().describe()
        data["model"] = self.model
        return data
