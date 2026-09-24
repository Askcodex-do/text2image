"""Face detection and identity comparison services.

Two responsibilities live here:

* :class:`FaceDetector` -- locate faces (used to build identity references and
  to drive face-region compositing).  Detection is *optional*: if no face is
  found the pipeline continues normally.
* :class:`IdentityVerifier` -- an optional, non-biometric post-generation
  comparison used purely as a quality-control signal.  It never claims to prove
  that two faces belong to the same individual.

The default detector uses OpenCV's bundled Haar cascades so the application
works with no extra downloads.  A provider or deployment may inject a stronger
detector (e.g. YuNet / an embedding model) without touching the pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..models import FaceBox, IdentityCheckResult


class FaceDetector:
    """Detects frontal faces using OpenCV Haar cascades.

    ``detect`` returns faces sorted largest-first so the "primary" subject is
    usually index 0.
    """

    def __init__(self, min_neighbors: int = 6, scale_factor: float = 1.1) -> None:
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self._cascade = cv2.CascadeClassifier(cascade_path)
        self._min_neighbors = min_neighbors
        self._scale_factor = scale_factor

    @property
    def available(self) -> bool:
        return not self._cascade.empty()

    def detect(self, image_path: str) -> list[FaceBox]:
        if not self.available:
            return []
        image = cv2.imread(image_path)
        if image is None:
            return []
        return self.detect_from_array(image)

    def detect_from_array(self, image: np.ndarray) -> list[FaceBox]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        regions = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=(40, 40),
        )
        boxes = [
            FaceBox(index=0, x=int(x), y=int(y), w=int(w), h=int(h), confidence=1.0)
            for (x, y, w, h) in regions
        ]
        boxes = _suppress_overlaps(boxes)
        boxes.sort(key=lambda b: b.area, reverse=True)
        for i, box in enumerate(boxes):
            box.index = i
        return boxes


def _suppress_overlaps(boxes: list[FaceBox], threshold: float = 0.35) -> list[FaceBox]:
    """Greedy non-maximum suppression by IoU.

    Haar cascades frequently emit several overlapping boxes for one face,
    especially at low ``minNeighbors``.  Suppressing overlaps keeps the
    multi-person UI ("Faces detected: N") honest.
    """
    kept: list[FaceBox] = []
    for box in sorted(boxes, key=lambda b: b.area, reverse=True):
        if all(_iou(box, other) < threshold for other in kept):
            kept.append(box)
    return kept


def _iou(a: FaceBox, b: FaceBox) -> float:
    x0 = max(a.x, b.x)
    y0 = max(a.y, b.y)
    x1 = min(a.x + a.w, b.x + b.w)
    y1 = min(a.y + a.h, b.y + b.h)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    union = a.area + b.area - intersection
    return intersection / union if union else 0.0


class IdentityVerifier:
    """Heuristic, non-biometric identity similarity comparison.

    The score is a quality-control signal built from two cheap, interpretable
    measurements on the detected face crops:

    * normalised cross-correlation of luminance (facial structure, eye/nose/
      mouth position and facial proportions) -- the dominant term
    * skin-tone agreement in HSV (hue and saturation similarity), which is
      brightness-invariant

    This is deliberately **not** a face-recognition model.  Calibration on the
    test fixtures separates the same face (~0.95+) from a different face
    (~0.2), but it cannot prove that two faces are the same individual, so the
    result is always marked ``biometric=False``.
    """

    #: Aggregate score at/above which the check is reported as completed.
    COMPLETED_THRESHOLD = 0.80
    #: Aggregate score at/above which the check asks for a light review.
    REVIEW_THRESHOLD = 0.60

    def __init__(self, detector: FaceDetector | None = None) -> None:
        self._detector = detector or FaceDetector()

    def compare(
        self,
        original_path: str,
        generated_path: str,
    ) -> IdentityCheckResult:
        original = cv2.imread(original_path)
        generated = cv2.imread(generated_path)
        if original is None or generated is None:
            return IdentityCheckResult(
                performed=False,
                verdict="not_performed",
                method="heuristic_v1",
                message="Could not read one of the images for comparison.",
            )

        orig_face = self._largest_face(original)
        gen_face = self._largest_face(generated)
        if orig_face is None:
            return IdentityCheckResult(
                performed=False,
                verdict="not_performed",
                method="heuristic_v1",
                message="No face detected in the original image; identity not compared.",
            )
        if gen_face is None:
            return IdentityCheckResult(
                performed=True,
                score=0.0,
                verdict="review_recommended",
                method="heuristic_v1",
                message="No face was detected in the generated image; review recommended.",
            )

        score, detail = self._similarity(
            self._crop(original, orig_face),
            self._crop(generated, gen_face),
        )
        if score >= self.COMPLETED_THRESHOLD:
            verdict = "completed"
            message = (
                "Identity preservation check completed. This is a non-biometric "
                "quality signal and not proof of identity."
            )
        elif score >= self.REVIEW_THRESHOLD:
            verdict = "review_recommended"
            message = "Identity preservation: review recommended."
        else:
            verdict = "review_recommended"
            message = (
                "Identity preservation: significant facial difference detected. "
                "Review recommended."
            )
        return IdentityCheckResult(
            performed=True,
            score=score,
            verdict=verdict,
            method="heuristic_v1",
            biometric=False,
            message=message,
        )

    def _largest_face(self, image: np.ndarray) -> FaceBox | None:
        faces = self._detector.detect_from_array(image)
        return faces[0] if faces else None

    @staticmethod
    def _crop(image: np.ndarray, face: FaceBox) -> np.ndarray:
        h, w = image.shape[:2]
        x0 = max(0, face.x)
        y0 = max(0, face.y)
        x1 = min(w, face.x + face.w)
        y1 = min(h, face.y + face.h)
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            return np.zeros((64, 64, 3), dtype=np.uint8)
        return cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _similarity(a: np.ndarray, b: np.ndarray) -> tuple[float, dict[str, float]]:
        """Return (score, component detail) for two aligned face crops."""
        ncc = _structural_correlation(a, b)
        tone = _skin_tone_similarity(a, b)
        # Structure dominates; skin tone nudges the score.  Both terms are in
        # [0, 1] (NCC is clamped from [-1, 1]).
        score = 0.8 * ncc + 0.2 * tone
        return max(0.0, min(1.0, score)), {"structural": ncc, "skin_tone": tone}


def _structural_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation of luminance, clamped to [0, 1]."""
    gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_a = (gray_a - gray_a.mean()) / (gray_a.std() + 1e-6)
    gray_b = (gray_b - gray_b.mean()) / (gray_b.std() + 1e-6)
    return max(0.0, min(1.0, float((gray_a * gray_b).mean())))


def _skin_tone_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Brightness-invariant hue/saturation agreement over non-extreme pixels."""
    hsv_a = cv2.cvtColor(a, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_b = cv2.cvtColor(b, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Restrict to mid-brightness pixels to avoid deep shadows and specular
    # highlights dominating the comparison.
    mask_a = (hsv_a[..., 2] > 30) & (hsv_a[..., 2] < 245)
    mask_b = (hsv_b[..., 2] > 30) & (hsv_b[..., 2] < 245)
    mean_a = hsv_a[mask_a] if mask_a.any() else hsv_a.reshape(-1, 3)
    mean_b = hsv_b[mask_b] if mask_b.any() else hsv_b.reshape(-1, 3)
    hue_diff = np.abs(mean_a[:, 0].mean() - mean_b[:, 0].mean())
    sat_diff = np.abs(mean_a[:, 1].mean() - mean_b[:, 1].mean())
    # Hue wraps at 180 in OpenCV; saturation spans 0-255.
    hue_score = 1.0 - min(hue_diff, 90.0) / 90.0
    sat_score = 1.0 - min(sat_diff, 128.0) / 128.0
    return float(max(0.0, min(1.0, 0.5 * hue_score + 0.5 * sat_score)))


@dataclass
class FaceDetectionResult:
    faces: list[FaceBox]
    detector_available: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "faces": [f.to_dict() for f in self.faces],
            "count": len(self.faces),
            "detector_available": self.detector_available,
        }