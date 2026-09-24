"""Remote HTTP image provider.

A generic client for an external image-generation API.  It is used in two ways:

* **Configured** (``IMAGE_API_URL`` + ``IMAGE_API_KEY`` set): real requests are
  sent to the backend.  If the backend advertises reference-image / identity
  support, the original image is sent as an identity reference.
* **Unconfigured**: the provider reports itself unavailable and the application
  tells the user honestly that identity preservation cannot be guaranteed.

Capability discovery is data-driven: the backend may expose a
``capabilities`` object, otherwise a conservative default set is assumed.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from typing import Any

from ..models import (
    GeneratedImage,
    GenerationRequest,
    PipelineContext,
    ProviderCapabilities,
)
from .base import ImageProvider, ProviderError, ProviderNotConfigured


class RemoteImageProvider(ImageProvider):
    name = "remote"
    label = "Remote API"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.api_url = (self.config.get("api_url") or os.environ.get("IMAGE_API_URL") or "").strip()
        self.api_key = (self.config.get("api_key") or os.environ.get("IMAGE_API_KEY") or "").strip()
        self.timeout = int(self.config.get("timeout") or os.environ.get("IMAGE_API_TIMEOUT") or 120)
        #: Optional cached capabilities reported by the remote service.
        self._remote_caps: dict[str, Any] | None = self.config.get("remote_capabilities")

    # -- Capabilities -----------------------------------------------------
    def capabilities(self) -> ProviderCapabilities:
        remote = self._remote_caps or {}
        identity = bool(remote.get("supports_face_preservation", False))
        return ProviderCapabilities(
            supports_face_preservation=identity,
            supports_reference_image=bool(remote.get("supports_reference_image", identity)),
            supports_multiple_faces=bool(remote.get("supports_multiple_faces", identity)),
            supports_composition_control=bool(remote.get("supports_composition_control", False)),
            supports_expression_control=bool(remote.get("supports_expression_control", False)),
            supports_identity_check=bool(remote.get("supports_identity_check", False)),
            supports_negative_prompt=bool(remote.get("supports_negative_prompt", True)),
            supports_seed=bool(remote.get("supports_seed", True)),
            is_remote=True,
            max_images_per_request=int(remote.get("max_images_per_request", 4)),
            strength_mapping=self._strength_mapping(identity),
            notes=[
                "Identity preservation depends on the remote backend's actual "
                "capabilities.",
                (
                    "This backend reports identity-reference support."
                    if identity
                    else "This backend does not report identity-preservation "
                    "support; the Preserve Face setting is best-effort only."
                ),
            ],
        )

    @staticmethod
    def _strength_mapping(identity: bool) -> dict[str, dict[str, Any]]:
        if not identity:
            return {}
        # Common diffusion-style control knobs; overridden by backend data.
        return {
            "low": {"identity_strength": 0.35, "ip_adapter_scale": 0.4},
            "medium": {"identity_strength": 0.6, "ip_adapter_scale": 0.65},
            "high": {"identity_strength": 0.8, "ip_adapter_scale": 0.85},
            "maximum": {"identity_strength": 1.0, "ip_adapter_scale": 1.0},
        }

    def is_available(self) -> bool:
        return bool(self.api_url)

    def unavailable_reason(self) -> str:
        if not self.api_url:
            return (
                "No remote backend is configured (set IMAGE_API_URL and "
                "IMAGE_API_KEY)."
            )
        return ""

    # -- Generation -------------------------------------------------------
    def generate(self, request: GenerationRequest, context: PipelineContext) -> list[GeneratedImage]:
        return self._call(request, context, edit=False)

    def edit(self, request: GenerationRequest, context: PipelineContext) -> list[GeneratedImage]:
        if not context.original_image:
            raise ProviderError("A remote edit requires an input image.")
        return self._call(request, context, edit=True)

    def _call(
        self,
        request: GenerationRequest,
        context: PipelineContext,
        edit: bool,
    ) -> list[GeneratedImage]:
        if not self.is_available():
            raise ProviderNotConfigured(self.unavailable_reason())

        caps = self.capabilities()
        payload: dict[str, Any] = {
            "prompt": context.effective_prompt,
            "style": request.style,
            "composition": request.composition,
            "aspect_ratio": request.aspect_ratio,
            "number_of_images": min(request.number_of_images, caps.max_images_per_request),
            "preserve_face": context.provider_uses_identity_reference,
            "face_preservation_strength": request.face_preservation_strength.value,
            "preserve_composition": request.preserve_composition,
            "preserve_expression": request.preserve_expression,
        }
        payload.update(context.provider_params)

        if edit and context.provider_uses_identity_reference:
            # The untouched original is sent as the identity reference.
            payload["reference_image"] = self._encode(context.original_image)
            payload["reference_image_role"] = "identity"
        elif edit:
            payload["image"] = self._encode(context.original_image)

        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        endpoint = self.api_url.rstrip("/") + ("/edit" if edit else "/generate")
        req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise ProviderError(f"Remote backend returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderError(f"Could not reach the remote backend: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ProviderError("The remote backend returned malformed JSON.") from exc

        return self._store_results(data, request, context)

    def _store_results(
        self,
        data: dict[str, Any],
        request: GenerationRequest,
        context: PipelineContext,
    ) -> list[GeneratedImage]:
        from ..services.storage import Storage

        storage: Storage = self.config["storage"]
        images = data.get("images") or data.get("data") or []
        if not images:
            raise ProviderError("The remote backend returned no images.")

        results: list[GeneratedImage] = []
        for index, item in enumerate(images):
            raw = item.get("b64_json") or item.get("image") or item.get("url") if isinstance(item, dict) else item
            if not raw:
                continue
            destination = storage.generated_path(
                original_filename=request.input_image or "image",
                style_key=request.style,
                provider_name=self.name,
                index=index,
            )
            self._write_bytes(raw, destination)
            results.append(
                GeneratedImage(
                    path=storage.relative_image_path(destination),
                    index=index,
                    original_path=(
                        storage.relative_image_path(context.original_image)
                        if context.original_image
                        else None
                    ),
                    seed=item.get("seed") if isinstance(item, dict) else None,
                    provider=self.name,
                    metadata={
                        "remote": True,
                        "identity_reference_sent": context.provider_uses_identity_reference,
                    },
                )
            )
        if not results:
            raise ProviderError("The remote backend returned no decodable images.")
        return results

    @staticmethod
    def _encode(path: str) -> str:
        with open(path, "rb") as handle:
            return base64.b64encode(handle.read()).decode("ascii")

    @staticmethod
    def _write_bytes(raw: str, destination: str) -> None:
        import cv2
        import numpy as np

        if raw.startswith("http://") or raw.startswith("https://"):
            with urllib.request.urlopen(raw) as response:  # noqa: S310 - configured backend
                payload = response.read()
        else:
            try:
                payload = base64.b64decode(raw, validate=True)
            except Exception as exc:  # noqa: BLE001
                raise ProviderError("The remote backend returned undecodable image data.") from exc
        array = np.frombuffer(payload, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise ProviderError("The remote backend returned image data that is not a valid image.")
        cv2.imwrite(destination, image)