"""Tests for frozen-executable (PyInstaller) runtime behaviour.

A frozen build unpacks into a temporary directory that is deleted on exit, so
these tests pin the behaviours the packaging depends on: outputs must go to a
writable persistent location, the face cascade must be findable outside
``sys._MEIPASS``, and browser auto-open must never be forced on source runs.
"""

from __future__ import annotations

import os
import sys

from ai_image_studio.config import AppConfig, default_output_dir
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


def test_frozen_config_enables_browser_open_but_source_does_not(monkeypatch, tmp_path):
    exe = tmp_path / "studio.exe"
    exe.write_bytes(b"stub")
    _simulate_frozen(monkeypatch, str(exe))
    assert AppConfig.from_env().open_browser is True

    monkeypatch.delattr(sys, "frozen")
    monkeypatch.delenv("AIS_OPEN_BROWSER", raising=False)
    assert AppConfig.from_env().open_browser is False


def test_browser_open_can_be_disabled_when_frozen(monkeypatch, tmp_path):
    exe = tmp_path / "studio.exe"
    exe.write_bytes(b"stub")
    _simulate_frozen(monkeypatch, str(exe))
    monkeypatch.setenv("AIS_OPEN_BROWSER", "0")
    assert AppConfig.from_env().open_browser is False


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