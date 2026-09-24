"""Application configuration."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

#: Subdirectory created next to the executable (frozen) or project (source).
_OUTPUT_DIR_NAME = "AIImageStudio"


def default_output_dir() -> str:
    """Return a writable output directory for both source and frozen runs.

    A PyInstaller bundle unpacks into a temporary directory that is deleted on
    exit, so outputs must never default there.  A frozen build writes next to
    the executable; a source run writes next to the project root.
    """
    override = os.environ.get("AIS_OUTPUT_DIR")
    if override:
        return override
    if getattr(sys, "frozen", False):
        base = os.path.dirname(os.path.abspath(sys.executable))
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, _OUTPUT_DIR_NAME)


def open_browser_enabled(default: bool) -> bool:
    """Read ``AIS_OPEN_BROWSER``, falling back to ``default``.

    Each entry point picks its own default: the windowed launcher opens a
    browser (`True`), while a console run stays in the terminal (`False`).
    """
    raw = os.environ.get("AIS_OPEN_BROWSER")
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


@dataclass
class AppConfig:
    """Runtime configuration, populated from environment variables."""

    output_dir: str = ""
    host: str = "0.0.0.0"
    port: int = 12000
    default_provider: str = "cloud"
    run_identity_check: bool = True
    debug: bool = False

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            output_dir=default_output_dir(),
            host=os.environ.get("AIS_HOST", "0.0.0.0"),
            port=int(os.environ.get("AIS_PORT", "12000")),
            default_provider=os.environ.get("AIS_PROVIDER", "cloud"),
            run_identity_check=os.environ.get("AIS_IDENTITY_CHECK", "1") != "0",
            debug=os.environ.get("AIS_DEBUG", "0") == "1",
        )

    @property
    def provider_config(self) -> dict:
        return {}  # extended by the app factory with the storage handle