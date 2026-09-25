"""Cloud image provider backed by a hosted text-to-image service.

This is the backend that actually *creates* what the user describes.  Unlike the
local OpenCV provider it interprets the prompt text, so requests such as
"chinese prince wearing a black robe and a gold crown" produce new content
rather than a filter over the original photo.

Two limitations are handled honestly rather than papered over:

* The hosted model accepts no reference image, so a specific person's face
  cannot be *conditioned* on.  Instead the user's own facial pixels are
  composited onto the generated result, which is genuine (if approximate)
  identity preservation.  :meth:`capabilities` reports this accurately, and the
  per-image metadata records what actually happened.
* The endpoint is deterministic for a given URL.  A distinct seed is therefore
  derived for every requested image, otherwise a 4-image batch would come back
  as four identical copies.

The endpoint and model are configurable (``IMAGE_GEN_URL`` / ``IMAGE_GEN_MODEL``)
so a different or self-hosted service can be dropped in without code changes.
"""

from __future__ import annotations

import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import numpy as np

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
        # The hosted service enforces a quota and rate limits bursts, so a batch
        # of images can fail partway.  Retry transient failures with backoff.
        self.retries = int(
            self.config.get("retries") or os.environ.get("IMAGE_GEN_RETRIES") or 2
        )
        self.retry_delay = float(
            self.config.get("retry_delay") or os.environ.get("IMAGE_GEN_RETRY_DELAY") or 3.0
        )
        self._detector = FaceDetector()

    # -- Capabilities -----------------------------------------------------
    def capabilities(self) -> ProviderCapabilities:
        # The hosted endpoint has no reference-image conditioning, but this
        # provider still preserves identity by a real mechanism: after
        # generation it detects the face in the result and composites the
        # user's own facial pixels onto it.  That is genuine preservation (the
        # person's pixels end up in the output), so it is advertised -- with an
        # honest note that it depends on the generator producing a comparable
        # head position.
        compositing = self._compositing_enabled()
        return ProviderCapabilities(
            supports_face_preservation=compositing,
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
            strength_mapping=(
                {
                    "low": {"face_blend": 0.40},
                    "medium": {"face_blend": 0.60},
                    "high": {"face_blend": 0.78},
                    "maximum": {"face_blend": 0.92},
                }
                if compositing
                else {}
            ),
            notes=[
                "Generates a brand-new image from your description, so it can "
                "create people, clothing and scenes that are not in the photo.",
                (
                    "The service does not accept a reference image, so the face "
                    "is preserved by compositing your original facial pixels "
                    "onto the generated result. This works best when the "
                    "generated head position is similar to the photo."
                )
                if compositing
                else (
                    "Face preservation is disabled for this provider; identity "
                    "is carried in the prompt only and is not guaranteed."
                ),
            ],
        )

    def _compositing_enabled(self) -> bool:
        """Whether to composite the original face onto generated output.

        On by default: without it, an uploaded photo contributes nothing but
        words, which is precisely the complaint that the app "does not use the
        face I gave it".  Set ``IMAGE_GEN_FACE_COMPOSITE=0`` to disable.
        """
        value = self.config.get("face_compositing")
        if value is None:
            value = os.environ.get("IMAGE_GEN_FACE_COMPOSITE", "1")
        if isinstance(value, str):
            return value.strip().lower() not in ("0", "false", "no", "off", "")
        return bool(value)

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

        # The hosted endpoint is deterministic: the same prompt without a seed
        # returns byte-identical images, so a 4-image batch used to be four
        # copies.  Always derive a distinct seed per output (the user's seed, if
        # given, anchors the sequence so results stay reproducible).
        base_seed = request.seed
        if base_seed is None:
            base_seed = int.from_bytes(os.urandom(4), "big") % 1_000_000

        results: list[GeneratedImage] = []
        for index in range(count):
            seed = int(base_seed) + index
            image_bytes = self._request_image(prompt, width, height, seed)
            results.append(
                self._store(image_bytes, request, context, index, seed, edit=edit)
            )
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

        last_error: ProviderError | None = None
        for attempt in range(self.retries + 1):
            if attempt:
                time.sleep(self.retry_delay * attempt)
            try:
                return self._fetch(url)
            except ProviderError as exc:
                last_error = exc
                if not self._is_transient(exc):
                    raise
        assert last_error is not None
        raise last_error

    def _fetch(self, url: str) -> bytes:
        request = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise ProviderError(
                self._explain_http_error(exc.code, detail)
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderError(f"Could not reach the generation service: {exc}") from exc
        if not payload:
            raise ProviderError("The generation service returned an empty response.")
        return payload

    @staticmethod
    def _explain_http_error(code: int, detail: str) -> str:
        """Turn a raw service error into something the user can act on."""
        if "INSUFFICIENT_BALANCE" in detail or "Insufficient balance" in detail:
            return (
                "The free generation quota for this service is exhausted, so no "
                "images were created. Wait a few minutes and try again, request "
                "fewer images at once, or switch to the Local provider. "
                "(Underlying error: insufficient balance.)"
            )
        if code == 429:
            return (
                "The generation service is rate-limiting requests. Wait a "
                "moment and try again, or request fewer images at once."
            )
        return f"The generation service returned HTTP {code}: {detail}"

    @staticmethod
    def _is_transient(exc: ProviderError) -> bool:
        """Whether retrying has any chance of succeeding."""
        text = str(exc)
        return (
            "quota" in text.lower()
            or "rate-limit" in text.lower()
            or "HTTP 5" in text
            or "Could not reach" in text
            or "empty response" in text
        )

    def _store(
        self,
        payload: bytes,
        request: GenerationRequest,
        context: PipelineContext,
        index: int,
        seed: int | None = None,
        edit: bool = False,
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

        preservation = "prompt-only (no reference image support)"
        if edit and request.identity_requested() and self._compositing_enabled():
            preservation = self._composite_faces(image, context)

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
            seed=seed,
            provider=self.name,
            metadata={
                "engine": "cloud",
                "model": self.model,
                "prompt": request.prompt,
                "identity_preservation": preservation,
            },
        )

    def _composite_faces(
        self,
        generated: np.ndarray,
        context: PipelineContext,
    ) -> str:
        """Composite the user's own facial pixels onto the generated image.

        The generator invents its own composition, so the original face is
        located in the *generated* frame and the original crop is resized onto
        it.  This is what makes "convert my photo" actually keep the person
        rather than producing a stranger who matches the description.
        """
        import cv2

        from ..services.face_composite import composite_original_face

        original = cv2.imread(context.original_image)
        if original is None:
            return "unavailable (original image could not be read)"

        selected = set(
            context.preserved_face_indices or [f.index for f in context.faces]
        )
        sources = [f for f in context.faces if f.index in selected]
        if not sources:
            return "skipped (no face detected in the original)"

        targets = self._detector.detect_from_array(generated)
        if not targets:
            return "skipped (no face found in the generated image)"

        face_blend = float(context.provider_params.get("face_blend", 0.78))
        applied = 0
        # Pair each source face with a generated face. The detector sorts
        # largest-first on both sides, so the primary subject usually lines up.
        for source_face, target_face in zip(sources, targets):
            if composite_original_face(
                generated, original, source_face, target_face, face_blend
            ):
                applied += 1

        if not applied:
            return "skipped (face region too small to composite)"
        return f"face-region compositing ({applied} face(s))"

    def describe(self) -> dict[str, Any]:
        data = super().describe()
        data["model"] = self.model
        return data
