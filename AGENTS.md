# AGENTS.md

## Project

AI Image Studio — Flask app for image generation/editing whose pipeline
separates IDENTITY from CONTENT from STYLE so a person's identity can be
preserved across a style transformation.

## Commands

- Run: `python run.py` (defaults to port 12000, honors `AIS_PORT`)
- Test: `python -m pytest tests/ -q`
- Install: `pip install -r requirements.txt`
- Build `.exe` (native Windows): `python -m PyInstaller packaging/ai_image_studio.spec --noconfirm`
- Build `.exe` (Linux cross-build via Wine): `bash packaging/build_exe.sh`
- Lint: `python -m pyflakes ai_image_studio tests`

## Packaging

- The frozen entry point is `run.py`; PyInstaller bundles `templates/` and
  `static/` as data. `app._resource_dir()` resolves them from `sys._MEIPASS`
  when frozen, so do not hardcode package-relative template paths.
- Output must never default inside the bundle: `config.default_output_dir()`
  writes beside `sys.executable` when frozen. `sys._MEIPASS` is deleted on exit.
- Face cascades are bundled explicitly and resolved through
  `face_service._cascade_candidates()`. Bundling is required because a frozen
  OpenCV may not expose `cv2.data.haarcascades`.
- Wine's `ucrtbase.dll` lacks `crealf`, so `numpy>=2` crashes under the
  cross-build. `packaging/build_exe.sh` pins `numpy==1.26.4` and
  `opencv-python-headless==4.10.0.84` for that reason only; native builds
  (`requirements.txt`) are unaffected.
- Do not commit build artifacts (`*.exe`, `dist/`, `build/`). Releases are
  produced by `.github/workflows/build-exe.yml`.

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