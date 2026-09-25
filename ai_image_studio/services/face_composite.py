"""Face-region compositing shared by the providers.

This is the mechanism that makes identity preservation *real* rather than a
prompt request: the person's own facial pixels are blended into the styled or
generated result through a feathered mask, so the style changes while the face
stays recognisable.

Two shapes of the problem are handled:

* :func:`blend_region_toward_original` -- the styled image and the original
  already line up pixel-for-pixel (the local spatial renderer, or a provider
  that honoured the composition).  The face is pulled *toward* the original.
* :func:`composite_original_face` -- the generated image has its own geometry
  (a text-to-image backend).  The original face crop is resized onto the
  generated face box, which is what lets a reference photo survive a generative
  style change.

The blend deliberately never fully reverts to the original: at maximum
strength the styled/generated pixels still contribute, otherwise a close-up
portrait would come back looking untouched.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..models import FaceBox


def feather_mask(shape: tuple[int, int], margin: float = 0.18) -> np.ndarray:
    """A soft-edged mask so the composited face has no visible seam."""
    height, width = shape
    mask = np.ones((height, width), dtype=np.float32)
    pad_y = max(1, int(height * margin))
    pad_x = max(1, int(width * margin))
    ramp_y = np.linspace(0.0, 1.0, pad_y, dtype=np.float32)
    ramp_x = np.linspace(0.0, 1.0, pad_x, dtype=np.float32)
    mask[:pad_y, :] *= ramp_y[:, None]
    mask[-pad_y:, :] *= ramp_y[::-1, None]
    mask[:, :pad_x] *= ramp_x[None, :]
    mask[:, -pad_x:] *= ramp_x[::-1][None, :]
    return mask


def expanded_box(
    face: FaceBox, width: int, height: int, pad_x_ratio: float = 0.25, pad_y_ratio: float = 0.30
) -> tuple[int, int, int, int] | None:
    """Face box grown to include the jaw and forehead the detector excludes.

    The detector is deliberately conservative, so compositing only its box
    would leave a styled halo around the chin and hairline.
    """
    pad_x = int(face.w * pad_x_ratio)
    pad_y = int(face.h * pad_y_ratio)
    x0 = max(0, face.x - pad_x)
    y0 = max(0, face.y - pad_y)
    x1 = min(width, face.x + face.w + pad_x)
    y1 = min(height, face.y + face.h + pad_y)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def original_face_weight(face_blend: float) -> float:
    """How strongly the original face replaces the styled/generated pixels.

    ``face_blend`` is the user-facing Face Preservation strength, so it must be
    monotonic: a higher setting means *more* of the user's own face, never less.
    The cap leaves the requested style visible even at Maximum, so a styled
    close-up does not come back looking untouched -- the specification asks the
    style to change while the person stays recognisable, not for a near-copy.
    """
    return float(np.clip(0.40 + 0.45 * float(face_blend), 0.0, 0.83))


def blend_region_toward_original(
    styled: np.ndarray,
    original: np.ndarray,
    box: tuple[int, int, int, int],
    face_blend: float,
) -> None:
    """Blend the original face *into* ``styled`` in place, preserving style."""
    x0, y0, x1, y1 = box
    face_region = original[y0:y1, x0:x1].astype(np.float32)
    styled_region = styled[y0:y1, x0:x1].astype(np.float32)
    mask = feather_mask(face_region.shape[:2])[..., None]
    weight = original_face_weight(face_blend)
    blended = face_region * (weight * mask) + styled_region * (1.0 - weight * mask)
    styled[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)


def composite_original_face(
    generated: np.ndarray,
    original: np.ndarray,
    source_face: FaceBox,
    target_face: FaceBox,
    face_blend: float,
) -> bool:
    """Resize the original face onto the generated image's face box.

    Used when the generator produced its own composition, so the two images do
    not line up.  Returns ``False`` when the region is unusable or too small to
    composite meaningfully.
    """
    height, width = generated.shape[:2]
    source_box = expanded_box(source_face, original.shape[1], original.shape[0])
    target_box = expanded_box(target_face, width, height)
    if source_box is None or target_box is None:
        return False

    sx0, sy0, sx1, sy1 = source_box
    tx0, ty0, tx1, ty1 = target_box
    crop = original[sy0:sy1, sx0:sx1]
    if crop.size == 0 or (tx1 - tx0) < 8 or (ty1 - ty0) < 8:
        return False

    resized = cv2.resize(crop, (tx1 - tx0, ty1 - ty0), interpolation=cv2.INTER_AREA)
    target = generated[ty0:ty1, tx0:tx1].astype(np.float32)
    mask = feather_mask((ty1 - ty0, tx1 - tx0), margin=0.22)[..., None]
    # ``face_blend`` pins identity; the generated pixels always keep a share so
    # the requested style still reads on the face.
    weight = original_face_weight(face_blend)
    blended = resized.astype(np.float32) * (weight * mask) + target * (1.0 - weight * mask)
    generated[ty0:ty1, tx0:tx1] = np.clip(blended, 0, 255).astype(np.uint8)
    return True
