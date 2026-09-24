"""HTTP API tests."""

from __future__ import annotations

import io
import os

import cv2


def upload(client, image_path, name="person.jpg"):
    with open(image_path, "rb") as handle:
        data = {"image": (io.BytesIO(handle.read()), name)}
    return client.post("/api/upload", data=data, content_type="multipart/form-data")


def test_index_renders_required_controls(client):
    body = client.get("/").get_data(as_text=True)
    for needle in (
        "AI Image Studio",
        "Describe what you want to create or change",
        "Face Preservation",
        "Preserve Face Identity",
        "Preserve Composition",
        "Preserve Expression",
        "Generated Images",
        "Style",
    ):
        assert needle in body, f"missing UI element: {needle}"
    # Tooltip required by the specification.
    assert "Attempts to preserve the person&#39;s facial identity" in body \
        or "Attempts to preserve the person's facial identity" in body


def test_config_exposes_capabilities(client):
    data = client.get("/api/config").get_json()
    providers = {p["name"]: p for p in data["providers"]}
    assert "local" in providers
    assert "cloud" in providers
    assert providers["local"]["supports_face_preservation"] is True
    assert providers["remote"]["supports_face_preservation"] is False
    # The local filter pipeline must not claim to interpret free text, and the
    # generative cloud backend must.
    assert providers["local"]["capabilities"]["honors_prompt"] is False
    assert providers["cloud"]["capabilities"]["honors_prompt"] is True
    # The generative backend is the default so prompts actually create images.
    assert data["default_provider"] == "cloud"
    assert data["face_preservation"]["levels"] == [
        "off", "low", "medium", "high", "maximum",
    ]
    assert data["face_preservation"]["default"] == "high"
    assert len(data["styles"]) >= 14


def test_upload_detects_faces_and_keeps_original(client, app, face_image):
    response = upload(client, face_image)
    data = response.get_json()
    assert response.status_code == 200
    assert data["face_count"] >= 1
    assert data["faces"][0]["w"] > 0

    # The user's source file is untouched.
    assert cv2.imread(face_image) is not None
    original_path = os.path.join(app.extensions["ais"]["storage"].root, data["image"]["path"])
    assert os.path.exists(original_path)


def test_upload_rejects_unknown_extension(client, face_image):
    response = upload(client, face_image, name="malware.exe")
    assert response.status_code == 400
    assert "Unsupported file type" in response.get_json()["error"]


def test_upload_rejects_non_image(client, tmp_path):
    bogus = tmp_path / "notanimage.png"
    bogus.write_bytes(b"this is definitely not a png")
    response = upload(client, str(bogus))
    assert response.status_code == 400


def test_upload_without_face_still_succeeds(client, blank_image):
    response = upload(client, blank_image, name="landscape.png")
    data = response.get_json()
    assert response.status_code == 200
    assert data["face_count"] == 0
    assert "No face detected" in data["message"]


def test_generate_edit_preserves_original(client, app, face_image):
    upload_response = upload(client, face_image)
    image_id = upload_response.get_json()["image"]["id"]
    before = open(face_image, "rb").read()

    response = client.post(
        "/api/generate",
        json={
            "prompt": "Convert this photograph into a realistic oil painting.",
            "input_image": image_id,
            "style": "oil_painting_realism",
            "composition": "portrait",
            "aspect_ratio": "original",
            "preserve_face": True,
            "face_preservation_strength": "high",
            "preserve_composition": True,
            "preserve_expression": True,
            "number_of_images": 1,
            "provider": "local",
        },
    )
    data = response.get_json()
    assert response.status_code == 200, data
    assert len(data["images"]) == 1
    assert data["pipeline"]["provider_uses_identity_reference"] is True
    assert data["pipeline"]["faces"]
    image = data["images"][0]
    assert image["original_path"] is not None
    assert "original" in image["original_path"]
    # Original bytes never change.
    assert open(face_image, "rb").read() == before

    # The generated file is served.
    media = client.get("/media/" + image["path"])
    assert media.status_code == 200


def test_generate_without_prompt_errors(client, face_image):
    image_id = upload(client, face_image).get_json()["image"]["id"]
    response = client.post(
        "/api/generate",
        json={"prompt": "  ", "input_image": image_id, "provider": "local"},
    )
    assert response.status_code == 400
    assert "description" in response.get_json()["error"].lower()


def test_generate_remote_provider_unavailable_is_honest(client, monkeypatch):
    monkeypatch.delenv("IMAGE_API_URL", raising=False)
    response = client.post(
        "/api/generate",
        json={"prompt": "a cat", "provider": "remote"},
    )
    assert response.status_code == 400
    assert "no remote backend is configured" in response.get_json()["error"].lower()


def test_preview_prompt_is_provider_specific(client, face_image):
    image_id = upload(client, face_image).get_json()["image"]["id"]
    payload = {
        "prompt": "Convert this photograph into a realistic oil painting.",
        "input_image": image_id,
        "style": "oil_painting_realism",
        "preserve_face": True,
        "face_preservation_strength": "high",
        "provider": "local",
    }
    local = client.post("/api/preview-prompt", json=payload).get_json()
    assert local["dialect"] == "stable_diffusion"
    assert local["uses_identity_reference"] is True
    assert local["clauses"]["identity"]
    assert local["effective_prompt"]

    payload["provider"] = "remote"
    remote = client.post("/api/preview-prompt", json=payload).get_json()
    assert remote["dialect"] == "openai"
    assert remote["uses_identity_reference"] is False
    assert remote["clauses"]["identity"] == ""
    assert any("best-effort" in w for w in remote["warnings"])
    assert remote["effective_prompt"] != local["effective_prompt"]


def test_preview_prompt_expression_override(client, face_image):
    image_id = upload(client, face_image).get_json()["image"]["id"]
    data = client.post(
        "/api/preview-prompt",
        json={
            "prompt": "Make the person smile naturally.",
            "input_image": image_id,
            "style": "oil_painting_realism",
            "preserve_face": True,
            "preserve_expression": True,
            "provider": "local",
        },
    ).get_json()
    assert data["clauses"]["expression"] == ""
    assert any("explicitly changes the expression" in w for w in data["warnings"])


def test_faces_endpoint(client, face_image):
    image_id = upload(client, face_image).get_json()["image"]["id"]
    data = client.post("/api/faces", json={"image_id": image_id}).get_json()
    assert data["count"] >= 1
    missing = client.post("/api/faces", json={"image_id": "does-not-exist"})
    assert missing.status_code == 404


def test_multi_person_selection(client, app, two_face_image):
    data = upload(client, two_face_image).get_json()
    assert data["face_count"] >= 2, "fixture must contain two detectable faces"
    image_id = data["image"]["id"]
    indices = [f["index"] for f in data["faces"]]

    response = client.post(
        "/api/generate",
        json={
            "prompt": "oil painting",
            "input_image": image_id,
            "style": "oil_painting_realism",
            "preserve_face": True,
            "face_preservation_strength": "maximum",
            "selected_face_indices": [indices[0]],
            "provider": "local",
            "number_of_images": 1,
        },
    )
    payload = response.get_json()
    assert response.status_code == 200, payload
    assert payload["pipeline"]["preserved_face_indices"] == [indices[0]]
    assert len(payload["pipeline"]["faces"]) >= 2


def test_media_path_traversal_blocked(client):
    response = client.get("/media/../../../../etc/passwd")
    assert response.status_code == 404