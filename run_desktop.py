#!/usr/bin/env python3
"""Windowed entry point for the desktop build: ``python run_desktop.py``.

Builds intended to be double-clicked use this instead of ``run.py``. It shows a
native window and opens the browser; see :mod:`ai_image_studio.desktop`.

A windowed executable has no console, so an unhandled exception would otherwise
vanish silently. Startup failures are therefore written to a log file and shown
in a message box.
"""
from __future__ import annotations

import os
import traceback


def _log_path() -> str:
    from ai_image_studio.config import default_output_dir

    directory = default_output_dir()
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, "launcher.log")


def _report_crash(message: str) -> None:
    try:
        with open(_log_path(), "a", encoding="utf-8") as handle:
            handle.write(message + "\n")
    except Exception:
        pass
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(  # type: ignore[attr-defined]
            None,
            "AI Image Studio could not start.\n\n"
            "Details were written to:\n"
            f"{_log_path()}\n\n"
            f"{message.splitlines()[-1] if message else ''}",
            "AI Image Studio",
            0x10,  # MB_ICONERROR
        )
    except Exception:
        pass


def main() -> int:
    try:
        from ai_image_studio.desktop import main as desktop_main

        return desktop_main()
    except SystemExit:
        raise
    except BaseException:
        _report_crash(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())