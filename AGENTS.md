# AGENTS.md

## Project

AI Image Studio — Flask app for image generation/editing whose pipeline
separates IDENTITY from CONTENT from STYLE so a person's identity can be
preserved across a style transformation.

## Commands

- Run: `python run.py` (defaults to port 12000, honors `AIS_PORT`)
- Test: `python -m pytest tests/ -q`
- Install: `pip install -r requirements.txt`

## Critical constraints

- `opencv-python-headless` MUST stay `<5`. OpenCV 5.0 removed
  `cv2.CascadeClassifier` and the Haar cascade data, which the face detector
  depends on.
- Originals in `AIImageStudio/originals/` are immutable. Never write to them.
  `Storage.is_original_path()` guards cleanup paths.
- The identity check is NON-BIOMETRIC (`IdentityCheckResult.biometric` is
  always False). Never present its score as proof of identity.
- A provider must never claim a capability it lacks. `supports_face_preservation`
  gates the GUI controls; when False the controls are disabled and a warning is
  recorded, never silently pretended.
- Prompt building is provider-specific (dialects: `stable_diffusion`, `openai`,
  `generic`). Do not introduce one universal prompt.
- Face detection is optional; absence of a face must never fail a request.

## Tests

`tests/assets/` holds small real face crops derived from OpenCV sample images
(Apache-2.0). Haar cascades do not detect synthetic drawings reliably, so keep
using real crops for detection tests.