"""Local, dependency-light image provider.

This provider performs real style transformations with OpenCV and is always
available -- no network or credentials required.

Identity preservation here is *genuine* rather than prompt-based: the original
face region is composited back over the stylised image through a feathered
mask, so the person's actual facial pixels (and therefore identity) survive the
style change.  Because the transform is spatial and local, pose, composition
and expression are inherently preserved too.

Colour/tone preservation for the face can be dialled by strength:
    low -> strong style modulation, face blended softly
    maximum -> original face kept almost untouched, blended tightly
"""

from __future__ import annotations

import cv2
import numpy as np

from ..models import (
    GeneratedImage,
    GenerationRequest,
    PipelineContext,
    ProviderCapabilities,
)
from .base import ImageProvider, ProviderError
from ..services.face_service import FaceDetector


def _variation_offsets(index: int, count: int) -> float:
    """Symmetric style-intensity offsets so a multi-image batch varies.

    With ``count`` images the offsets span roughly -0.2 .. +0.2.
    """
    if count <= 1:
        return 0.0
    return 0.4 * (index / (count - 1) - 0.5)


class LocalStyleProvider(ImageProvider):
    name = "local"
    label = "Local (OpenCV)"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self._detector = FaceDetector()

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_face_preservation=True,
            supports_reference_image=True,
            supports_multiple_faces=True,
            supports_composition_control=False,
            supports_expression_control=True,
            supports_identity_check=True,
            supports_negative_prompt=False,
            supports_seed=True,
            is_remote=False,
            max_images_per_request=8,
            strength_mapping={
                "low": {"face_blend": 0.55, "style_strength": 1.0},
                "medium": {"face_blend": 0.75, "style_strength": 0.85},
                "high": {"face_blend": 0.88, "style_strength": 0.7},
                "maximum": {"face_blend": 0.96, "style_strength": 0.5},
            },
            notes=[
                "Local rendering preserves identity by keeping the original "
                "face pixels and styling the rest of the image.",
                "Face preservation works best on clearly visible, frontal "
                "faces.",
            ],
        )

    def is_available(self) -> bool:
        return True

    def generate(self, request: GenerationRequest, context: PipelineContext) -> list[GeneratedImage]:
        raise ProviderError(
            "The local provider renders stylised edits of an uploaded image. "
            "Upload a photo to use it."
        )

    def edit(self, request: GenerationRequest, context: PipelineContext) -> list[GeneratedImage]:
        if not context.original_image:
            raise ProviderError("The local provider requires an input image.")
        source = cv2.imread(context.original_image)
        if source is None:
            raise ProviderError("Could not read the input image.")

        params = context.provider_params
        base_style_strength = float(params.get("style_strength", 0.75))
        face_blend = float(params.get("face_blend", 0.88))
        preserve_face = request.identity_requested() and bool(context.faces)

        # Render one image per requested output, varying the style intensity
        # slightly so the set is genuinely different rather than duplicated.
        count = max(1, min(request.number_of_images, 8))
        results: list[GeneratedImage] = []
        for index in range(count):
            variation = _variation_offsets(index, count)
            style_strength = float(
                np.clip(base_style_strength + variation, 0.2, 1.0)
            )
            output = self._apply_style(source, request.style, style_strength)
            if preserve_face:
                output = self._composite_faces(output, source, context, face_blend)
            results.append(self._write(output, context, request, index))
        return results

    # -- Styling ----------------------------------------------------------
    def _apply_style(self, image: np.ndarray, style: str, strength: float) -> np.ndarray:
        image = image.copy()
        if style == "black_and_white":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if style == "pencil_sketch":
            return self._pencil(image, strength)
        if style == "charcoal":
            return self._pencil(image, min(1.0, strength + 0.2), dark=True)
        if style == "oil_painting":
            return self._oil(image, strength, strong=True)
        if style in ("oil_painting_realism", "renaissance", "cinematic", "concept_art"):
            return self._oil(image, strength)
        if style == "watercolor":
            return self._watercolor(image, strength)
        if style == "anime":
            return self._anime(image, strength)
        if style == "cartoon":
            return self._cartoon(image, strength)
        if style == "vintage":
            return self._vintage(image, strength)
        if style == "digital_art":
            return self._digital(image, strength)
        if style == "fantasy":
            return self._fantasy(image, strength)
        return self._oil(image, strength * 0.5)

    def _oil(self, image: np.ndarray, strength: float, strong: bool = False) -> np.ndarray:
        radius = 8 if strong else 5
        styled = cv2.stylization(image, sigma_s=int(20 + strength * 40), sigma_r=0.45)
        styled = cv2.bilateralFilter(styled, d=radius, sigmaColor=80, sigmaSpace=80)
        return self._blend(image, styled, 0.5 + 0.5 * strength)

    def _pencil(self, image: np.ndarray, strength: float, dark: bool = False) -> np.ndarray:
        gray, color = cv2.pencilSketch(
            image, sigma_s=60, sigma_r=0.07, shade_factor=0.05
        )
        base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if dark:
            base = cv2.convertScaleAbs(base, alpha=0.75, beta=-10)
        return self._blend(image, base, 0.6 + 0.4 * strength)

    def _watercolor(self, image: np.ndarray, strength: float) -> np.ndarray:
        styled = cv2.stylization(image, sigma_s=60, sigma_r=0.25)
        styled = cv2.bilateralFilter(styled, d=9, sigmaColor=120, sigmaSpace=120)
        bright = cv2.convertScaleAbs(styled, alpha=1.08, beta=12)
        return self._blend(image, bright, 0.5 + 0.5 * strength)

    def _anime(self, image: np.ndarray, strength: float) -> np.ndarray:
        smooth = cv2.bilateralFilter(image, d=11, sigmaColor=120, sigmaSpace=120)
        smooth = cv2.stylization(smooth, sigma_s=80, sigma_r=0.35)
        edges = cv2.adaptiveThreshold(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 255,
            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9,
        )
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        saturated = cv2.convertScaleAbs(smooth, alpha=1.25, beta=10)
        cel = cv2.bitwise_and(saturated, edges)
        return self._blend(image, cel, 0.55 + 0.45 * strength)

    def _cartoon(self, image: np.ndarray, strength: float) -> np.ndarray:
        smooth = cv2.bilateralFilter(image, d=15, sigmaColor=140, sigmaSpace=140)
        quantised = (smooth // 32) * 32 + 16
        edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 80, 180)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        shapes = cv2.bitwise_and(quantised, cv2.bitwise_not(edges))
        return self._blend(image, shapes, 0.5 + 0.5 * strength)

    def _vintage(self, image: np.ndarray, strength: float) -> np.ndarray:
        sepia_kernel = np.array(
            [[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]
        )
        sepia = cv2.transform(image, sepia_kernel)
        sepia = np.clip(sepia, 0, 255).astype(np.uint8)
        faded = cv2.convertScaleAbs(sepia, alpha=0.9, beta=18)
        noise = np.random.default_rng(1234).normal(0, 6, faded.shape)
        grainy = np.clip(faded.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return self._blend(image, grainy, 0.4 + 0.6 * strength)

    def _digital(self, image: np.ndarray, strength: float) -> np.ndarray:
        smooth = cv2.bilateralFilter(image, d=9, sigmaColor=90, sigmaSpace=90)
        vivid = cv2.convertScaleAbs(smooth, alpha=1.15, beta=5)
        hsv = cv2.cvtColor(vivid, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * (1.1 + 0.3 * strength), 0, 255)
        return self._blend(image, cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), 0.5 + 0.5 * strength)

    def _fantasy(self, image: np.ndarray, strength: float) -> np.ndarray:
        stylised = cv2.stylization(image, sigma_s=70, sigma_r=0.4)
        glow = cv2.GaussianBlur(stylised, (0, 0), sigmaX=6)
        glowed = cv2.addWeighted(stylised, 0.75, glow, 0.35, 8)
        return self._blend(image, glowed, 0.5 + 0.5 * strength)

    @staticmethod
    def _blend(original: np.ndarray, styled: np.ndarray, amount: float) -> np.ndarray:
        amount = float(np.clip(amount, 0.0, 1.0))
        return cv2.addWeighted(styled, amount, original, 1.0 - amount, 0)

    # -- Face-aware compositing ------------------------------------------
    def _composite_faces(
        self,
        styled: np.ndarray,
        original: np.ndarray,
        context: PipelineContext,
        face_blend: float,
    ) -> np.ndarray:
        """Blend the original face region back over the stylised image.

        This is the local provider's real identity-preservation mechanism.
        """
        result = styled.copy()
        selected = set(context.preserved_face_indices or [f.index for f in context.faces])
        height, width = original.shape[:2]
        for face in context.faces:
            if face.index not in selected:
                continue
            # Expand the crop slightly to include the jaw/forehead that the
            # detector typically excludes, then feather the seam.
            pad_x = int(face.w * 0.25)
            pad_y = int(face.h * 0.30)
            x0 = max(0, face.x - pad_x)
            y0 = max(0, face.y - pad_y)
            x1 = min(width, face.x + face.w + pad_x)
            y1 = min(height, face.y + face.h + pad_y)
            if x1 <= x0 or y1 <= y0:
                continue
            face_region = original[y0:y1, x0:x1]
            styled_region = result[y0:y1, x0:x1]
            mask = self._feather_mask(face_region.shape[:2])[..., None]
            blended = (
                face_region.astype(np.float32) * (face_blend * mask)
                + styled_region.astype(np.float32) * (1.0 - face_blend * mask)
            )
            result[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
        return result

    @staticmethod
    def _feather_mask(shape: tuple[int, int], margin: float = 0.18) -> np.ndarray:
        height, width = shape
        mask = np.ones((height, width), dtype=np.float32)
        pad_y = max(1, int(height * margin))
        pad_x = max(1, int(width * margin))
        # Linear ramps on each edge produce a soft, artefact-free blend.
        ramp_y = np.linspace(0.0, 1.0, pad_y, dtype=np.float32)
        ramp_x = np.linspace(0.0, 1.0, pad_x, dtype=np.float32)
        mask[:pad_y, :] *= ramp_y[:, None]
        mask[-pad_y:, :] *= ramp_y[::-1, None]
        mask[:, :pad_x] *= ramp_x[None, :]
        mask[:, -pad_x:] *= ramp_x[::-1][None, :]
        return mask

    def _write(
        self,
        image: np.ndarray,
        context: PipelineContext,
        request: GenerationRequest,
        index: int,
    ) -> GeneratedImage:
        from ..services.storage import Storage

        storage: Storage = self.config["storage"]
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
            original_path=storage.relative_image_path(context.original_image),
            provider=self.name,
            metadata={
                "engine": "opencv",
                "style": request.style,
                "face_preservation": (
                    "face-region compositing"
                    if context.provider_uses_identity_reference
                    else "none"
                ),
            },
        )