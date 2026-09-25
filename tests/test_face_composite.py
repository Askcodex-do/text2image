"""Tests for the shared face-compositing helpers.

The central invariant: a higher Face Preservation strength must put *more* of
the user's own face into the output.  The original implementation had this
inverted, which is why raising the setting made the given face matter less.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai_image_studio.models import FaceBox
from ai_image_studio.services.face_composite import (
    blend_region_toward_original,
    composite_original_face,
    expanded_box,
    feather_mask,
    original_face_weight,
)


def _solid(value: int, height: int = 64, width: int = 64) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_weight_is_monotonic_in_strength():
    weights = [original_face_weight(b) for b in (0.3, 0.5, 0.7, 0.9)]
    assert weights == sorted(weights), "more strength must mean more original face"
    assert weights[-1] <= 0.98, "style must remain visible even at Maximum"


def test_region_blend_approaches_original_as_strength_rises():
    styled = _solid(200)
    original = _solid(50)
    box = (0, 0, 64, 64)

    centers = []
    for strength in (0.3, 0.6, 0.9):
        target = styled.copy()
        blend_region_toward_original(target, original, box, strength)
        centers.append(int(target[32, 32, 0]))
    # Lower pixel value == closer to the (dark) original face.
    assert centers == sorted(centers, reverse=True), centers


def test_composite_drives_output_toward_the_original_face():
    generated = _solid(220, 120, 120)
    original = _solid(40, 120, 120)
    source = FaceBox(index=0, x=20, y=20, w=60, h=60)
    target = FaceBox(index=0, x=20, y=20, w=60, h=60)

    assert composite_original_face(generated, original, source, target, 0.92)
    assert generated[50, 50, 0] < 120, "the original face must darken the centre"


def test_expanded_box_is_clamped_to_the_frame():
    face = FaceBox(index=0, x=2, y=3, w=10, h=10)
    box = expanded_box(face, width=40, height=40)
    assert box is not None
    x0, y0, x1, y1 = box
    assert 0 <= x0 < x1 <= 40
    assert 0 <= y0 < y1 <= 40


def test_feather_mask_edges_are_soft_and_centre_is_opaque():
    mask = feather_mask((80, 80))
    assert mask[40, 40] == pytest.approx(1.0)
    assert mask[0, 40] == pytest.approx(0.0, abs=1e-6)
    assert mask[40, 0] == pytest.approx(0.0, abs=1e-6)
