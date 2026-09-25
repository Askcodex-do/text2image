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
from ..services.face_composite import blend_region_toward_original, expanded_box, feather_mask


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
            supports_seed=False,
            # The local renderer applies a preset OpenCV style; it does not read
            # the prompt text, so it must not claim to.
            honors_prompt=False,
            is_remote=False,
            max_images_per_request=8,
            strength_mapping={
                "low": {"face_blend": 0.35, "style_strength": 1.0},
                "medium": {"face_blend": 0.55, "style_strength": 0.9},
                "high": {"face_blend": 0.75, "style_strength": 0.8},
                "maximum": {"face_blend": 0.98, "style_strength": 0.7},
            },
            notes=[
                "Local rendering applies the selected style to your whole photo "
                "and blends the original face back to keep the person "
                "recognisable.",
                "This backend does not interpret the prompt text and cannot add "
                "new people, objects or clothing. Use the Cloud provider for that.",
                "Face preservation works best on clearly visible, frontal faces.",
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
            return self._pencil(image, min(1.0, strength + 0.3), dark=True)
        if style == "oil_painting":
            return self._oil(image, strength)
        if style == "oil_painting_realism":
            return self._oil_realism(image, strength)
        if style == "renaissance":
            return self._renaissance(image, strength)
        if style == "cinematic":
            return self._cinematic(image, strength)
        if style == "concept_art":
            return self._concept_art(image, strength)
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
        return self._oil(image, strength * 0.75)

    def _oil(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Visible brushwork: heavy edge-preserving smoothing plus a dappled
        texture layer, so the painterly effect is unmistakable."""
        styled = cv2.stylization(image, sigma_s=int(30 + strength * 70), sigma_r=0.35)
        styled = cv2.bilateralFilter(styled, d=9, sigmaColor=90, sigmaSpace=90)
        styled = self._brush_texture(styled, strength)
        return self._blend(image, styled, 0.75 + 0.25 * strength)

    def _oil_realism(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Painterly but lifelike: lighter smoothing and a gentle texture."""
        styled = cv2.stylization(image, sigma_s=int(20 + strength * 35), sigma_r=0.45)
        styled = cv2.bilateralFilter(styled, d=7, sigmaColor=70, sigmaSpace=70)
        styled = self._brush_texture(styled, strength * 0.45)
        return self._blend(image, styled, 0.6 + 0.3 * strength)

    def _renaissance(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Chiaroscuro: deep shadows, warm earth tones, visible brushwork."""
        styled = cv2.stylization(image, sigma_s=int(35 + strength * 60), sigma_r=0.3)
        styled = self._brush_texture(styled, strength)
        warm = cv2.transform(
            styled, np.array([[0.72, 0.42, 0.16], [0.38, 0.68, 0.14], [0.24, 0.56, 0.20]])
        )
        warm = np.clip(warm, 0, 255).astype(np.uint8)
        # Lift contrast for the dramatic light/shadow of the period.
        contrast = cv2.convertScaleAbs(warm, alpha=1.25, beta=-28)
        return self._blend(image, contrast, 0.8 + 0.2 * strength)

    def _cinematic(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Teal-and-orange grade with lifted contrast and a soft vignette."""
        smooth = cv2.bilateralFilter(image, d=7, sigmaColor=60, sigmaSpace=60)
        lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[..., 1] = np.clip(lab[..., 1] + 14 * strength, 0, 255)   # +green/magenta
        lab[..., 2] = np.clip(lab[..., 2] + 18 * strength, 0, 255)   # +blue/yellow
        graded = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        graded = cv2.convertScaleAbs(graded, alpha=1.2, beta=-14)
        graded = self._vignette(graded, 0.55)
        return self._blend(image, graded, 0.78 + 0.22 * strength)

    def _concept_art(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Bold painterly rendering: strong smoothing, high contrast, vivid colour."""
        styled = cv2.stylization(image, sigma_s=int(45 + strength * 60), sigma_r=0.3)
        styled = self._brush_texture(styled, strength)
        vivid = cv2.convertScaleAbs(styled, alpha=1.25, beta=-12)
        hsv = cv2.cvtColor(vivid, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * (1.2 + 0.4 * strength), 0, 255)
        vivid = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return self._blend(image, vivid, 0.8 + 0.2 * strength)

    @staticmethod
    def _brush_texture(image: np.ndarray, strength: float) -> np.ndarray:
        """Overlay a fixed-seed dapple that reads as brush/impasto strokes."""
        if strength <= 0:
            return image
        h, w = image.shape[:2]
        rng = np.random.default_rng(20240517)
        noise = rng.normal(128.0, 26.0, (h, w)).astype(np.float32)
        strokes = cv2.GaussianBlur(noise, (0, 0), sigmaX=2.2)[..., None]
        amount = 0.16 * float(np.clip(strength, 0.0, 1.0))
        textured = image.astype(np.float32) * (1.0 - amount) + strokes * amount
        return np.clip(textured, 0, 255).astype(np.uint8)

    @staticmethod
    def _vignette(image: np.ndarray, amount: float) -> np.ndarray:
        h, w = image.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        radius /= max(radius.max(), 1.0)
        mask = np.clip(1.0 - amount * radius**2, 0.0, 1.0)[..., None]
        return np.clip(image.astype(np.float32) * mask, 0, 255).astype(np.uint8)

    def _pencil(self, image: np.ndarray, strength: float, dark: bool = False) -> np.ndarray:
        gray, color = cv2.pencilSketch(
            image, sigma_s=60, sigma_r=0.07, shade_factor=0.05
        )
        base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if dark:
            base = cv2.convertScaleAbs(base, alpha=0.7, beta=-18)
        return self._blend(image, base, 0.75 + 0.25 * strength)

    def _watercolor(self, image: np.ndarray, strength: float) -> np.ndarray:
        styled = cv2.stylization(image, sigma_s=70, sigma_r=0.2)
        styled = cv2.bilateralFilter(styled, d=11, sigmaColor=140, sigmaSpace=140)
        bright = cv2.convertScaleAbs(styled, alpha=1.12, beta=18)
        return self._blend(image, bright, 0.7 + 0.3 * strength)

    def _anime(self, image: np.ndarray, strength: float) -> np.ndarray:
        smooth = cv2.bilateralFilter(image, d=11, sigmaColor=120, sigmaSpace=120)
        smooth = cv2.stylization(smooth, sigma_s=80, sigma_r=0.35)
        edges = cv2.adaptiveThreshold(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 255,
            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9,
        )
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        saturated = cv2.convertScaleAbs(smooth, alpha=1.3, beta=12)
        cel = cv2.bitwise_and(saturated, edges)
        return self._blend(image, cel, 0.7 + 0.3 * strength)

    def _cartoon(self, image: np.ndarray, strength: float) -> np.ndarray:
        smooth = cv2.bilateralFilter(image, d=15, sigmaColor=140, sigmaSpace=140)
        quantised = (smooth // 32) * 32 + 16
        edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 80, 180)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        shapes = cv2.bitwise_and(quantised, cv2.bitwise_not(edges))
        return self._blend(image, shapes, 0.75 + 0.25 * strength)

    def _vintage(self, image: np.ndarray, strength: float) -> np.ndarray:
        sepia_kernel = np.array(
            [[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]
        )
        sepia = cv2.transform(image, sepia_kernel)
        sepia = np.clip(sepia, 0, 255).astype(np.uint8)
        faded = cv2.convertScaleAbs(sepia, alpha=0.9, beta=18)
        noise = np.random.default_rng(1234).normal(0, 6, faded.shape)
        grainy = np.clip(faded.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return self._blend(image, grainy, 0.7 + 0.3 * strength)

    def _digital(self, image: np.ndarray, strength: float) -> np.ndarray:
        smooth = cv2.bilateralFilter(image, d=9, sigmaColor=90, sigmaSpace=90)
        vivid = cv2.convertScaleAbs(smooth, alpha=1.15, beta=5)
        hsv = cv2.cvtColor(vivid, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * (1.25 + 0.4 * strength), 0, 255)
        return self._blend(
            image, cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR),
            0.7 + 0.3 * strength,
        )

    def _fantasy(self, image: np.ndarray, strength: float) -> np.ndarray:
        stylised = cv2.stylization(image, sigma_s=70, sigma_r=0.4)
        glow = cv2.GaussianBlur(stylised, (0, 0), sigmaX=6)
        glowed = cv2.addWeighted(stylised, 0.8, glow, 0.45, 12)
        return self._blend(image, glowed, 0.7 + 0.3 * strength)

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
        """Reinforce identity by blending the *styled* face back toward the
        original through a feathered mask.

        The face must still receive the requested style -- copying the original
        face in wholesale was what previously made a portrait look unchanged.
        ``face_blend`` is how strongly the original facial structure is pulled
        back over the stylised pixels; the styling is never undone completely.
        """
        result = styled.copy()
        selected = set(context.preserved_face_indices or [f.index for f in context.faces])
        height, width = original.shape[:2]
        for face in context.faces:
            if face.index not in selected:
                continue
            box = expanded_box(face, width, height)
            if box is None:
                continue
            blend_region_toward_original(result, original, box, face_blend)
        return result

    @staticmethod
    def _feather_mask(shape: tuple[int, int], margin: float = 0.18) -> np.ndarray:
        return feather_mask(shape, margin)

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
                "identity_preservation": (
                    "face-region compositing"
                    if context.identity_preservation_active
                    else "none"
                ),
            },
        )