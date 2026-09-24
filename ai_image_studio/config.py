"""Application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class AppConfig:
    """Runtime configuration, populated from environment variables."""

    output_dir: str = ""
    host: str = "0.0.0.0"
    port: int = 12000
    default_provider: str = "local"
    run_identity_check: bool = True
    debug: bool = False

    @classmethod
    def from_env(cls) -> "AppConfig":
        default_output = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "AIImageStudio",
        )
        return cls(
            output_dir=os.environ.get("AIS_OUTPUT_DIR", default_output),
            host=os.environ.get("AIS_HOST", "0.0.0.0"),
            port=int(os.environ.get("AIS_PORT", "12000")),
            default_provider=os.environ.get("AIS_PROVIDER", "local"),
            run_identity_check=os.environ.get("AIS_IDENTITY_CHECK", "1") != "0",
            debug=os.environ.get("AIS_DEBUG", "0") == "1",
        )

    @property
    def provider_config(self) -> dict:
        return {}  # extended by the app factory with the storage handle