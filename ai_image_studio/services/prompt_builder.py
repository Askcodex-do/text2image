"""Prompt builder.

Critical design point from the specification: there is **no single universal
prompt**.  Each provider declares its own prompt dialect, and the builder
assembles only the clauses that provider understands.

The builder always keeps identity, content and style as separate clauses so a
provider capable of true reference-image conditioning can rely on the actual
reference image rather than a textual "keep the face the same" instruction.
"""

from __future__ import annotations

from typing import Any

from ..models import FacePreservationStrength, GenerationRequest, PipelineContext
from ..styles import STYLES

BASE_IDENTITY_INSTRUCTION = (
    "Preserve the identity and recognizable facial characteristics of the "
    "person in the reference image."
)

COMPOSITION_INSTRUCTION = "Preserve the original composition where possible."

EXPRESSION_INSTRUCTION = (
    "Preserve the person's original facial expression."
)

STYLISE_WARNING_CLAUSE = (
    "The visual style should change while the person remains recognizable."
)


def _style_label(style_key: str) -> str:
    preset = STYLES.get(style_key)
    return preset.label if preset else style_key


def _collect_user_expression_override(prompt: str) -> bool:
    """Detect an explicit request to change expression.

    The user's explicit instruction takes precedence over ``preserve_expression``
    (specification section 10).
    """
    lowered = (prompt or "").lower()
    markers = (
        "smile", "smiling", "laughing", "frown", "serious expression",
        "change expression", "neutral expression", "surprised", "angry",
        "sad expression", "make them smile", "look happy", "look sad",
        "look angry", "look surprised", "grin",
    )
    return any(marker in lowered for marker in markers)


class PromptBuilder:
    """Builds provider-specific prompts from a request and pipeline context."""

    #: Provider dialect identifier; subclasses/instances override this.
    dialect = "generic"

    def __init__(self, dialect: str = "generic") -> None:
        self.dialect = dialect

    def build(
        self,
        request: GenerationRequest,
        context: PipelineContext,
        provider_caps: Any = None,
    ) -> PipelineContext:
        """Populate prompt clauses and the effective prompt on ``context``."""
        identity_allowed = (
            request.identity_requested()
            and getattr(provider_caps, "supports_face_preservation", False)
            and context.faces
        )
        if request.identity_requested() and not context.faces:
            context.warnings.append(
                "No face was detected in the original image, so identity "
                "preservation was skipped."
            )
        if request.identity_requested() and not getattr(
            provider_caps, "supports_face_preservation", False
        ):
            if getattr(provider_caps, "honors_prompt", False):
                context.warnings.append(
                    "This provider does not use your photo as an identity "
                    "reference, so the Preserve Face setting is best-effort: the "
                    "subject is described in the prompt and a close likeness is "
                    "not guaranteed."
                )
            else:
                context.warnings.append(
                    "The selected provider cannot preserve identity: it neither "
                    "uses a reference image nor interprets the prompt. The "
                    "Preserve Face setting has no effect."
                )

        expression_override = _collect_user_expression_override(request.prompt)
        preserve_expression = (
            request.preserve_expression and identity_allowed and not expression_override
        )
        if request.preserve_expression and expression_override:
            context.warnings.append(
                "Your prompt explicitly changes the expression, so expression "
                "preservation was not applied."
            )

        context.style_prompt = self._style_clause(request)
        context.identity_prompt = (
            self._identity_clause(request, context) if identity_allowed else ""
        )
        context.composition_prompt = (
            self._composition_clause(request, provider_caps)
            if (request.preserve_composition or request.composition not in ("", "original"))
            else ""
        )
        context.expression_prompt = (
            EXPRESSION_INSTRUCTION if preserve_expression else ""
        )
        context.effective_prompt = self._assemble(request, context)
        context.provider_params.update(
            self._provider_params(request, provider_caps, identity_allowed)
        )
        return context

    # -- Clauses ----------------------------------------------------------
    def _style_clause(self, request: GenerationRequest) -> str:
        if not request.style or request.style == "none":
            return ""
        preset = STYLES.get(request.style)
        if not preset:
            return f"Apply a {request.style} style."
        if self.dialect == "stable_diffusion":
            return f"{preset.description} ({preset.label} style)."
        if self.dialect == "openai":
            return f"Restyle the image as {preset.label.lower()}. {preset.description}"
        return f"Style: {preset.label}. {preset.description}"

    def _identity_clause(self, request: GenerationRequest, context: PipelineContext) -> str:
        strength = request.face_preservation_strength
        if strength is FacePreservationStrength.MAXIMUM:
            extra = (
                " Keep the exact same person: identical facial structure, eye "
                "shape and position, nose, mouth, facial proportions, skin tone "
                "and any beard or moustache."
            )
        elif strength is FacePreservationStrength.HIGH:
            extra = (
                " Keep the same facial structure, facial proportions and "
                "distinguishing features."
            )
        elif strength is FacePreservationStrength.MEDIUM:
            extra = " Keep the face clearly recognisable."
        else:
            extra = " Keep the person recognisable."
        clause = BASE_IDENTITY_INSTRUCTION + extra
        if len(context.preserved_face_indices) > 1:
            clause += (
                f" There are {len(context.preserved_face_indices)} people in the "
                "image; preserve every person's identity."
            )
        return clause

    def _composition_clause(self, request: GenerationRequest, provider_caps: Any) -> str:
        if request.composition == "original" or not request.composition:
            return COMPOSITION_INSTRUCTION
        from ..styles import COMPOSITIONS

        preset = COMPOSITIONS.get(request.composition)
        if not preset:
            return COMPOSITION_INSTRUCTION
        return COMPOSITION_INSTRUCTION + f" Framing: {preset.description}"

    # -- Assembly ---------------------------------------------------------
    def _assemble(self, request: GenerationRequest, context: PipelineContext) -> str:
        parts: list[str] = []
        if request.prompt:
            parts.append(request.prompt.strip())
        if context.style_prompt:
            parts.append(context.style_prompt)
        if context.identity_prompt:
            parts.append(context.identity_prompt)
        if context.composition_prompt:
            parts.append(context.composition_prompt)
        if context.expression_prompt:
            parts.append(context.expression_prompt)
        if context.identity_prompt:
            parts.append(STYLISE_WARNING_CLAUSE)
        separator = ", " if self.dialect == "stable_diffusion" else " "
        return separator.join(p.strip().rstrip(".,") for p in parts if p.strip())

    # -- Provider params --------------------------------------------------
    def _provider_params(
        self,
        request: GenerationRequest,
        provider_caps: Any,
        identity_allowed: bool,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        caps_ok = getattr(provider_caps, "supports_face_preservation", False)
        if identity_allowed and caps_ok:
            mapping = getattr(provider_caps, "strength_mapping", {}) or {}
            params.update(mapping.get(request.face_preservation_strength.value, {}))
        if request.seed is not None and getattr(provider_caps, "supports_seed", False):
            params["seed"] = request.seed
        if request.negative_prompt and getattr(provider_caps, "supports_negative_prompt", False):
            params["negative_prompt"] = request.negative_prompt
        return params


class StableDiffusionPromptBuilder(PromptBuilder):
    def __init__(self) -> None:
        super().__init__(dialect="stable_diffusion")


class OpenAIPromptBuilder(PromptBuilder):
    def __init__(self) -> None:
        super().__init__(dialect="openai")


class GenericPromptBuilder(PromptBuilder):
    def __init__(self) -> None:
        super().__init__(dialect="generic")


DIALECTS = {
    "generic": GenericPromptBuilder,
    "stable_diffusion": StableDiffusionPromptBuilder,
    "openai": OpenAIPromptBuilder,
}


def builder_for(dialect: str) -> PromptBuilder:
    return DIALECTS.get(dialect, GenericPromptBuilder)()