"""Shared test fixtures.

Face-related tests use small cropped portraits built from OpenCV's standard
sample images (Apache-2.0 licensed, tracked in ``tests/assets``).  Haar
cascades detect real photographic faces far more reliably than synthetic ones,
so using real crops exercises the true detection code path.

The no-face path uses a generated high-contrast rectangle image.
"""

from __future__ import annotations

import os
import sys
import tempfile

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_image_studio.config import AppConfig  # noqa: E402
from ai_image_studio.app import create_app  # noqa: E402

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


@pytest.fixture()
def output_dir():
    with tempfile.TemporaryDirectory() as directory:
        yield directory


@pytest.fixture()
def asset_dir():
    return ASSETS


def make_blank_image(path: str, width: int = 300, height: int = 200) -> str:
    """A high-contrast but faceless image, so detection returns nothing."""
    image = np.full((height, width, 3), 128, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (width - 20, height - 20), (60, 120, 180), 4)
    cv2.line(image, (20, 20), (width - 20, height - 20), (230, 230, 230), 3)
    cv2.putText(image, "NO FACE", (60, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (20, 20, 20), 3)
    cv2.imwrite(path, image)
    return path


@pytest.fixture()
def face_image(tmp_path):
    source = os.path.join(ASSETS, "test_face.jpg")
    destination = str(tmp_path / "person.jpg")
    cv2.imwrite(destination, cv2.imread(source))
    return destination


@pytest.fixture()
def two_face_image(tmp_path):
    source = os.path.join(ASSETS, "test_two_faces.jpg")
    destination = str(tmp_path / "people.jpg")
    cv2.imwrite(destination, cv2.imread(source))
    return destination


@pytest.fixture()
def blank_image(tmp_path):
    return make_blank_image(str(tmp_path / "landscape.png"))


@pytest.fixture()
def app(output_dir):
    config = AppConfig(output_dir=output_dir)
    application = create_app(config)
    application.config.update(TESTING=True)
    return application


@pytest.fixture()
def client(app):
    return app.test_client()