"""Tests for frozen-executable (PyInstaller) runtime behaviour.

A frozen build unpacks into a temporary directory that is deleted on exit, so
these tests pin the behaviours the packaging depends on: outputs must go to a
writable persistent location, the face cascade must be findable outside
``sys._MEIPASS``, and browser auto-open must never be forced on source runs.
"""

from __future__ import annotations

import os
import sys

from ai_image_studio.config import default_output_dir, open_browser_enabled
from ai_image_studio.services.face_service import FaceDetector, _cascade_candidates


def _simulate_frozen(monkeypatch, executable: str) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", executable)
    monkeypatch.delenv("AIS_OUTPUT_DIR", raising=False)


def test_output_dir_uses_executable_location_when_frozen(monkeypatch, tmp_path):
    exe = tmp_path / "dist" / "AI Image Studio.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"stub")
    _simulate_frozen(monkeypatch, str(exe))

    out = default_output_dir()
    # Must sit beside the executable, never inside the temporary bundle.
    assert out == os.path.join(str(exe.parent), "AIImageStudio")
    assert "_MEI" not in out


def test_output_dir_ignores_bundle_temp_when_meipass_set(monkeypatch, tmp_path):
    exe = tmp_path / "app" / "studio.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"stub")
    _simulate_frozen(monkeypatch, str(exe))
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "_MEI12345"), raising=False)

    out = default_output_dir()
    assert out.startswith(str(exe.parent))
    assert "_MEI12345" not in out


def test_env_override_still_wins_when_frozen(monkeypatch, tmp_path):
    exe = tmp_path / "studio.exe"
    exe.write_bytes(b"stub")
    _simulate_frozen(monkeypatch, str(exe))
    chosen = tmp_path / "custom-out"
    monkeypatch.setenv("AIS_OUTPUT_DIR", str(chosen))
    assert default_output_dir() == str(chosen)


def test_browser_open_defaults_are_per_entry_point(monkeypatch):
    """The windowed launcher opens a browser; a console run does not."""
    monkeypatch.delenv("AIS_OPEN_BROWSER", raising=False)
    assert open_browser_enabled(default=True) is True   # desktop launcher
    assert open_browser_enabled(default=False) is False  # console run

    # The environment variable always wins over the per-entry-point default.
    monkeypatch.setenv("AIS_OPEN_BROWSER", "0")
    assert open_browser_enabled(default=True) is False
    monkeypatch.setenv("AIS_OPEN_BROWSER", "1")
    assert open_browser_enabled(default=False) is True
    # Falsy spellings are accepted so scripts can disable it readably.
    for value in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("AIS_OPEN_BROWSER", value)
        assert open_browser_enabled(default=True) is False


def test_cascade_candidates_include_bundle_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "bundle"), raising=False)
    candidates = _cascade_candidates()
    assert len(candidates) >= 2
    assert candidates[1] == os.path.join(
        str(tmp_path / "bundle"), "cv2", "data", "haarcascade_frontalface_default.xml"
    )


def test_detector_available_and_finds_a_real_face(face_image):
    """The cascade must load and detect through whatever path is resolvable."""
    detector = FaceDetector()
    assert detector.available, "front face cascade failed to load"
    assert detector.detect(face_image)


def test_detector_without_cascade_degrades_gracefully(monkeypatch):
    """A missing cascade must not raise; detection simply reports nothing."""
    monkeypatch.setattr(
        "ai_image_studio.services.face_service._cascade_candidates", lambda: []
    )
    detector = FaceDetector()
    assert detector.available is False
    assert detector.detect("does-not-exist.png") == []


# --- windowed (GUI) build configuration -------------------------------------
# The shipped artifact is a windowed executable. These assertions read the build
# inputs directly so a regression back to a console build fails fast in CI,
# instead of only showing up as a stray console window on a user's machine.

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC_PATH = os.path.join(PROJECT_ROOT, "packaging", "ai_image_studio.spec")
WORKFLOW_PATH = os.path.join(PROJECT_ROOT, ".github", "workflows", "build-exe.yml")


def _spec_source() -> str:
    with open(SPEC_PATH, encoding="utf-8") as handle:
        return handle.read()


def test_spec_builds_a_windowed_executable():
    spec = _spec_source()
    # console defaults to False (windowed) unless AIS_CONSOLE is set at build time.
    assert "console=CONSOLE" in spec
    assert 'os.environ.get("AIS_CONSOLE", "0")' in spec


def test_spec_uses_the_windowed_entry_point():
    spec = _spec_source()
    assert "run_desktop.py" in spec
    assert "run.py" not in spec.replace("run_desktop.py", "")


def test_spec_bundles_tkinter_for_the_lazy_import():
    """``desktop`` imports tkinter inside a function, so it needs hidden imports."""
    spec = _spec_source()
    for module in ("tkinter", "tkinter.ttk", "tkinter.messagebox"):
        assert f'"{module}"' in spec


def test_ci_pins_the_target_python_and_asserts_a_gui_subsystem():
    with open(WORKFLOW_PATH, encoding="utf-8") as handle:
        workflow = handle.read()
    assert "3.10.11" in workflow
    # The CI step must fail if the artifact is a console binary (subsystem 3).
    assert '-ne "2"' in workflow
    assert "subsystem" in workflow