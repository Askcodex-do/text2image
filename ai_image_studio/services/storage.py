"""Storage layer.

Hard rule: **originals are immutable**.  An upload is copied into
``originals/`` and never written to again.  Every generated image lands in
``images/`` with a collision-safe, human-readable name:

    photos/person.jpg                    <- user's file, untouched
    AIImageStudio/originals/<hash>.png   <- immutable working copy
    AIImageStudio/images/person_oil_painting_001.png
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import time
from dataclasses import dataclass
from typing import Any

import cv2

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


class StorageError(RuntimeError):
    pass


@dataclass
class StoredOriginal:
    id: str
    path: str
    filename: str
    width: int
    height: int
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "filename": self.filename,
            "width": self.width,
            "height": self.height,
            "size_bytes": self.size_bytes,
        }


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text or "").strip("_").lower()
    return cleaned or "image"


class Storage:
    """Filesystem-backed store rooted at ``root``."""

    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        self.originals_dir = os.path.join(self.root, "originals")
        self.images_dir = os.path.join(self.root, "images")
        self.uploads_dir = os.path.join(self.root, "uploads")
        for directory in (self.originals_dir, self.images_dir, self.uploads_dir):
            os.makedirs(directory, exist_ok=True)

    # -- Originals --------------------------------------------------------
    def store_upload(self, source_path: str, original_filename: str) -> StoredOriginal:
        """Validate and copy an uploaded file into immutable storage.

        The source file is only read; it is never modified.
        """
        ext = os.path.splitext(original_filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise StorageError(
                f"Unsupported file type '{ext or original_filename}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
            )
        size = os.path.getsize(source_path)
        if size > MAX_UPLOAD_BYTES:
            raise StorageError(
                f"File is {size / 1e6:.1f} MB, exceeding the "
                f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB limit."
            )

        image = cv2.imread(source_path)
        if image is None:
            raise StorageError("The uploaded file is not a readable image.")

        with open(source_path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        stem = _slug(os.path.splitext(os.path.basename(original_filename))[0])
        image_id = f"{stem}_{digest[:12]}"
        dest = os.path.join(self.originals_dir, f"{image_id}.png")
        if not os.path.exists(dest):
            # Normalise to PNG without ever touching the source file.
            cv2.imwrite(dest, image)

        height, width = image.shape[:2]
        return StoredOriginal(
            id=image_id,
            path=dest,
            filename=original_filename,
            width=width,
            height=height,
            size_bytes=size,
        )

    def original_path(self, image_id: str) -> str | None:
        if not image_id or "/" in image_id or "\\" in image_id or ".." in image_id:
            return None
        candidate = os.path.join(self.originals_dir, f"{image_id}.png")
        return candidate if os.path.exists(candidate) else None

    def is_original_path(self, path: str) -> bool:
        """True when ``path`` points inside the immutable originals directory."""
        try:
            resolved = os.path.abspath(path)
            return os.path.commonpath([resolved, self.originals_dir]) == self.originals_dir
        except (ValueError, TypeError):
            return False

    # -- Generated images -------------------------------------------------
    def generated_path(
        self,
        original_filename: str | None,
        style_key: str,
        provider_name: str,
        index: int,
        suffix: str = ".png",
    ) -> str:
        stem = _slug(os.path.splitext(os.path.basename(original_filename or "image"))[0])
        # Drop the content-hash suffix that immutable-original ids carry so the
        # generated filename stays human-readable (person_oil_painting_001.png).
        stem = re.sub(r"_[0-9a-f]{12}$", "", stem) or "image"
        style_part = _slug(style_key)
        base = f"{stem}_{style_part}_{provider_name}"
        pattern = re.compile(rf"^{re.escape(base)}_(\d{{3}}){re.escape(suffix)}$")
        highest = 0
        for name in os.listdir(self.images_dir):
            match = pattern.match(name)
            if match:
                highest = max(highest, int(match.group(1)))
        number = max(highest + 1, index + 1)
        return os.path.join(self.images_dir, f"{base}_{number:03d}{suffix}")

    def relative_image_path(self, path: str) -> str:
        return os.path.relpath(path, self.root).replace(os.sep, "/")

    def resolve_image(self, relative_path: str) -> str | None:
        """Resolve a client-supplied relative path inside the storage root."""
        if not relative_path:
            return None
        candidate = os.path.abspath(os.path.join(self.root, relative_path))
        try:
            if os.path.commonpath([candidate, self.root]) != self.root:
                return None
        except ValueError:
            return None
        return candidate if os.path.exists(candidate) else None

    def save_temp_upload(self, data: bytes, filename: str) -> str:
        os.makedirs(self.uploads_dir, exist_ok=True)
        safe = _slug(os.path.splitext(filename)[0])
        path = os.path.join(
            self.uploads_dir, f"{safe}_{int(time.time() * 1000)}"
            f"{os.path.splitext(filename)[1].lower() or '.png'}"
        )
        with open(path, "wb") as handle:
            handle.write(data)
        return path

    def cleanup_temp(self, path: str) -> None:
        """Remove a staging file (never an original)."""
        try:
            if path and self.is_original_path(path):
                return
            if path and os.path.exists(path):
                os.remove(path)
        except OSError:
            pass

    def copy_original(self, source: str, destination: str) -> str:
        """Copy the immutable original to a working destination."""
        shutil.copyfile(source, destination)
        return destination