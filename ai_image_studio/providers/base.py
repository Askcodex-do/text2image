"""Provider abstraction.

Every backend must declare honestly what it supports via
:class:`~ai_image_studio.models.ProviderCapabilities`.  The GUI enables or
disables controls based on those flags and never pretends a control has an
effect the backend cannot deliver.

``supports_face_preservation()`` is the single source of truth for whether
identity preservation is *actually available* for a provider.
"""

from __future__ import annotations

import abc
from typing import Any

from ..models import (
    GeneratedImage,
    GenerationRequest,
    PipelineContext,
    ProviderCapabilities,
)


class ProviderError(RuntimeError):
    """Raised when a provider cannot fulfil a request."""


class ProviderNotConfigured(ProviderError):
    """Raised when a remote provider is selected but has no credentials."""


class ImageProvider(abc.ABC):
    """Base class for all image backends."""

    #: Stable identifier used in API payloads and on-disk metadata.
    name: str = "provider"
    #: Human-readable label for the UI.
    label: str = "Provider"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    # -- Capabilities -----------------------------------------------------
    @abc.abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Return the precise capabilities of this backend."""

    def supports_face_preservation(self) -> bool:
        """Whether this backend can genuinely help preserve identity.

        Defaults to the capability flag so subclasses that support real
        reference-image identity preservation only need to override
        :meth:`capabilities`.
        """
        return self.capabilities().supports_face_preservation

    def is_available(self) -> bool:
        """Whether the provider can currently be used (e.g. credentials)."""
        return True

    def unavailable_reason(self) -> str:
        return ""

    # -- Generation -------------------------------------------------------
    @abc.abstractmethod
    def generate(self, request: GenerationRequest, context: PipelineContext) -> list[GeneratedImage]:
        """Produce images for a text-to-image request."""

    @abc.abstractmethod
    def edit(self, request: GenerationRequest, context: PipelineContext) -> list[GeneratedImage]:
        """Produce images for an image-edit request.

        Implementations receive ``context.original_image`` -- the untouched
        source photograph -- and should use it as an identity reference when
        they are capable of doing so.
        """

    def describe(self) -> dict[str, Any]:
        caps = self.capabilities()
        return {
            "name": self.name,
            "label": self.label,
            "available": self.is_available(),
            "unavailable_reason": self.unavailable_reason(),
            "capabilities": caps.to_dict(),
            "supports_face_preservation": self.supports_face_preservation(),
        }