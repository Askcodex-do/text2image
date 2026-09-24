"""Core data models for AI Image Studio.

These models deliberately separate *identity*, *content*, and *style* concerns
so that the pipeline can reason about each independently:

    Identity  -> what makes the person recognisable (preserved when requested)
    Content   -> geometry of the scene: pose, composition, background layout
    Style     -> the requested visual transformation (always allowed to change)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, asdict
from typing import Any


class FacePreservationStrength(str, enum.Enum):
    """Strength of identity preservation requested by the user.

    The concrete effect of each level is provider-defined (see
    ``ProviderCapabilities.strength_mapping``).  ``OFF`` disables the feature
    regardless of the provider's own support.
    """

    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"

    @classmethod
    def from_value(cls, value: Any) -> "FacePreservationStrength":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.HIGH
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            return cls.HIGH

    @property
    def rank(self) -> int:
        return _STRENGTH_ORDER.index(self)


_STRENGTH_ORDER = [
    FacePreservationStrength.OFF,
    FacePreservationStrength.LOW,
    FacePreservationStrength.MEDIUM,
    FacePreservationStrength.HIGH,
    FacePreservationStrength.MAXIMUM,
]


@dataclass
class FaceBox:
    """A detected face region in the original image (pixel coordinates)."""

    index: int
    x: int
    y: int
    w: int
    h: int
    confidence: float = 1.0
    #: Optional per-face descriptor produced by the detector; used by the
    #: identity verifier.  Kept opaque so detectors can evolve.
    descriptor: dict[str, Any] = field(default_factory=dict)

    @property
    def area(self) -> int:
        return self.w * self.h

    def to_dict(self, include_descriptor: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "index": self.index,
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "confidence": round(self.confidence, 4),
        }
        if include_descriptor:
            data["descriptor"] = self.descriptor
        return data


@dataclass
class ProviderCapabilities:
    """What a backend actually supports.

    The GUI uses these flags to decide which controls are meaningful.  A
    provider must never claim a capability it cannot honour: when
    ``supports_face_preservation`` is ``False`` the application tells the user
    that identity preservation is best-effort only.
    """

    supports_face_preservation: bool = False
    supports_reference_image: bool = False
    supports_multiple_faces: bool = False
    supports_composition_control: bool = False
    supports_expression_control: bool = False
    supports_identity_check: bool = False
    supports_negative_prompt: bool = False
    supports_seed: bool = False
    is_remote: bool = False
    max_images_per_request: int = 4
    #: Provider-specific mapping from strength level -> provider parameters.
    strength_mapping: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Human-readable notes shown in the UI (honesty about limitations).
    notes: list[str] = field(default_factory=list)

    def supported_strengths(self) -> list[str]:
        if not self.supports_face_preservation:
            return [FacePreservationStrength.OFF.value]
        mapping = self.strength_mapping or {}
        levels = [s.value for s in _STRENGTH_ORDER if s is not FacePreservationStrength.OFF]
        return [level for level in levels if level in mapping] or levels

    def strength_params(self, strength: FacePreservationStrength) -> dict[str, Any]:
        return dict((self.strength_mapping or {}).get(strength.value, {}))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["supported_strengths"] = self.supported_strengths()
        return data


@dataclass
class GenerationRequest:
    """A single user generation/edit request.

    Mirrors the request model in the specification.  Fields that a provider
    cannot honour are ignored by that provider but still recorded so the
    pipeline can explain *why* a control had no effect.
    """

    prompt: str
    input_image: str | None = None
    style: str = "none"
    composition: str = "original"
    aspect_ratio: str = "original"

    preserve_face: bool = True
    face_preservation_strength: FacePreservationStrength = FacePreservationStrength.HIGH

    preserve_composition: bool = False
    preserve_expression: bool = False

    number_of_images: int = 1

    negative_prompt: str | None = None
    seed: int | None = None

    #: Restrict preservation to a subset of detected faces (multi-person).
    selected_face_indices: list[int] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerationRequest":
        strength = FacePreservationStrength.from_value(data.get("face_preservation_strength"))
        return cls(
            prompt=str(data.get("prompt") or "").strip(),
            input_image=data.get("input_image"),
            style=str(data.get("style") or "none"),
            composition=str(data.get("composition") or "original"),
            aspect_ratio=str(data.get("aspect_ratio") or "original"),
            preserve_face=bool(data.get("preserve_face", True)),
            face_preservation_strength=strength,
            preserve_composition=bool(data.get("preserve_composition", False)),
            preserve_expression=bool(data.get("preserve_expression", False)),
            number_of_images=max(1, int(data.get("number_of_images", 1) or 1)),
            negative_prompt=data.get("negative_prompt"),
            seed=data.get("seed"),
            selected_face_indices=data.get("selected_face_indices"),
        )

    @property
    def is_edit(self) -> bool:
        return bool(self.input_image)

    def identity_requested(self) -> bool:
        return self.preserve_face and self.face_preservation_strength is not FacePreservationStrength.OFF

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["face_preservation_strength"] = self.face_preservation_strength.value
        data["is_edit"] = self.is_edit
        return data


@dataclass
class IdentityCheckResult:
    """Result of an optional post-generation identity comparison.

    ``biometric`` is ``False`` for the built-in heuristic verifier: the score
    is a *quality-control signal*, not proof that the generated person is the
    same individual.
    """

    performed: bool
    score: float | None = None
    verdict: str = "not_performed"
    method: str = ""
    biometric: bool = False
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data["score"] is not None:
            data["score"] = round(float(data["score"]), 4)
        return data


@dataclass
class GeneratedImage:
    """A produced image on disk plus its provenance metadata."""

    path: str
    index: int
    original_path: str | None = None
    seed: int | None = None
    provider: str = ""
    identity_check: IdentityCheckResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "path": self.path,
            "index": self.index,
            "original_path": self.original_path,
            "seed": self.seed,
            "provider": self.provider,
            "metadata": self.metadata,
        }
        data["identity_check"] = (
            self.identity_check.to_dict() if self.identity_check else None
        )
        return data


@dataclass
class PipelineContext:
    """Everything derived from the original image for a single request.

    The ``original_image`` path is carried through the entire request so any
    provider capable of true reference-image identity preservation can use it.
    """

    original_image: str | None
    faces: list[FaceBox] = field(default_factory=list)
    style_prompt: str = ""
    identity_prompt: str = ""
    composition_prompt: str = ""
    expression_prompt: str = ""
    effective_prompt: str = ""
    provider_params: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    #: Faces the user asked to preserve (defaults to all detected faces).
    preserved_face_indices: list[int] = field(default_factory=list)
    provider_uses_identity_reference: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_image": self.original_image,
            "faces": [f.to_dict() for f in self.faces],
            "style_prompt": self.style_prompt,
            "identity_prompt": self.identity_prompt,
            "composition_prompt": self.composition_prompt,
            "expression_prompt": self.expression_prompt,
            "effective_prompt": self.effective_prompt,
            "provider_params": self.provider_params,
            "warnings": list(self.warnings),
            "preserved_face_indices": list(self.preserved_face_indices),
            "provider_uses_identity_reference": self.provider_uses_identity_reference,
        }