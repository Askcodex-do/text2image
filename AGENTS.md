# AGENTS.md

## Project

AI Image Studio — Flask app for image generation/editing whose pipeline
separates IDENTITY from CONTENT from STYLE so a person's identity can be
preserved across a style transformation.

## Commands

- Run (browser GUI): `python run.py` (defaults to port 12000, honors `AIS_PORT`)
- Run (windowed launcher): `python run_desktop.py`
- Test: `python -m pytest tests/ -q`
- Install: `pip install -r requirements.txt`
- Build `.exe` (native Windows): `python -m PyInstaller packaging/ai_image_studio.spec --noconfirm`
- Build `.exe` (Linux cross-build via Wine): `bash packaging/build_exe.sh`
- Build console `.exe` instead of the windowed one: `AIS_CONSOLE=1 pyinstaller packaging/ai_image_studio.spec --noconfirm`
- Lint: `python -m pyflakes ai_image_studio tests`

## Packaging

- The frozen entry point is `run_desktop.py` (windowed). `run.py` remains the
  console/browser entry point for source runs. PyInstaller bundles `templates/`
  and `static/` as data; `app._resource_dir()` resolves them from `sys._MEIPASS`
  when frozen, so do not hardcode package-relative template paths.
- The build targets **Python 3.10.11**. The window requires Tcl/Tk, which is
  absent from the embeddable zip and the NuGet package; only the full CPython
  installer ships it. `packaging/build_exe.sh` therefore extracts the
  installer's MSI payload (`a0`=core, `a2`=exe, `a6`=Lib, `a14`=tkinter+tcl/tk)
  rather than using the embeddable distribution.
- The spec ships a **windowed** executable (`console=False`, PE subsystem 2):
  `desktop.run_windowed` shows a Tk window, and closing it calls `quit_()`,
  which shuts the server down. A windowed build has no stderr, so the launcher
  logs to `AIImageStudio/app.log`. `tkinter` is imported lazily, so it is listed
  in `hiddenimports` for PyInstaller's static analysis.
- A second launch is refused by `single_instance.InstanceLock` and simply opens
  the running instance's URL in the browser.
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
  produced by `.github/workflows/build-exe.yml`, which pins Python 3.10.11 and
  asserts the artifact's PE subsystem is 2 (windowed) before releasing.

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