"""Provider registry."""

from __future__ import annotations

from typing import Any

from .base import ImageProvider
from .local_provider import LocalStyleProvider
from .remote_provider import RemoteImageProvider

PROVIDERS: dict[str, type[ImageProvider]] = {
    LocalStyleProvider.name: LocalStyleProvider,
    RemoteImageProvider.name: RemoteImageProvider,
}

DEFAULT_PROVIDER = LocalStyleProvider.name


def create_provider(name: str | None, config: dict[str, Any] | None = None) -> ImageProvider:
    key = (name or DEFAULT_PROVIDER).strip().lower()
    factory = PROVIDERS.get(key) or LocalStyleProvider
    return factory(config)


def describe_providers(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [factory(config).describe() for factory in PROVIDERS.values()]