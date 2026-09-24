"""Flask application factory and HTTP API for AI Image Studio."""

from __future__ import annotations

import os
from typing import Any

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_from_directory,
)

from .config import AppConfig
from .models import FacePreservationStrength, GenerationRequest, PipelineContext
from .providers import create_provider, describe_providers
from .services import ImagePipeline, PipelineError, Storage, StorageError
from .styles import ASPECT_RATIOS, DEFAULT_STYLE, composition_choices, style_choices
from .services.prompt_builder import builder_for
from .services.pipeline import PROVIDER_DIALECTS


def create_app(config: AppConfig | None = None) -> Flask:
    config = config or AppConfig.from_env()
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024

    storage = Storage(config.output_dir)
    pipeline = ImagePipeline(
        storage=storage,
        run_identity_check=config.run_identity_check,
    )

    def provider_config() -> dict[str, Any]:
        return {"storage": storage}

    def get_provider(name: str):
        return create_provider(name or config.default_provider, provider_config())

    # -- Pages ------------------------------------------------------------
    @app.route("/")
    def index():
        return render_template(
            "index.html",
            styles=style_choices(),
            compositions=composition_choices(),
            aspect_ratios=ASPECT_RATIOS,
            default_style=DEFAULT_STYLE,
            providers=describe_providers(provider_config()),
            default_provider=config.default_provider,
        )

    @app.route("/media/<path:relative_path>")
    def media(relative_path: str):
        path = storage.resolve_image(relative_path)
        if not path:
            return jsonify({"error": "Not found"}), 404
        directory, filename = os.path.split(path)
        return send_from_directory(directory, filename)

    # -- API --------------------------------------------------------------
    @app.route("/api/config")
    def api_config():
        """Capabilities and option lists the GUI needs to render itself."""
        return jsonify(
            {
                "providers": describe_providers(provider_config()),
                "default_provider": config.default_provider,
                "styles": style_choices(),
                "compositions": composition_choices(),
                "aspect_ratios": ASPECT_RATIOS,
                "face_preservation": {
                    "levels": [s.value for s in FacePreservationStrength],
                    "default": FacePreservationStrength.HIGH.value,
                },
                "identity_check_enabled": config.run_identity_check,
            }
        )

    @app.route("/api/upload", methods=["POST"])
    def api_upload():
        if "image" not in request.files:
            return jsonify({"error": "No image file was provided."}), 400
        file = request.files["image"]
        if not file.filename:
            return jsonify({"error": "The uploaded file has no name."}), 400

        temp_path = storage.save_temp_upload(file.read(), file.filename)
        try:
            stored = storage.store_upload(temp_path, file.filename)
        except StorageError as exc:
            return jsonify({"error": str(exc)}), 400
        finally:
            storage.cleanup_temp(temp_path)

        faces = pipeline.detect_faces(stored.id)
        return jsonify(
            {
                "image": stored.to_dict(),
                "url": f"/media/originals/{stored.id}.png",
                "faces": faces,
                "face_count": len(faces),
                "detector_available": pipeline.detector.available,
                "message": (
                    f"{len(faces)} face(s) detected."
                    if faces
                    else "No face detected. Identity preservation will be skipped."
                ),
            }
        )

    @app.route("/api/faces", methods=["POST"])
    def api_faces():
        data = request.get_json(silent=True) or {}
        image_id = data.get("image_id")
        if not image_id:
            return jsonify({"error": "image_id is required."}), 400
        try:
            faces = pipeline.detect_faces(image_id)
        except PipelineError as exc:
            return jsonify({"error": str(exc)}), 404
        return jsonify(
            {
                "faces": faces,
                "count": len(faces),
                "detector_available": pipeline.detector.available,
            }
        )

    @app.route("/api/preview-prompt", methods=["POST"])
    def api_preview_prompt():
        """Show exactly what will be sent to the selected provider."""
        data = request.get_json(silent=True) or {}
        req = GenerationRequest.from_dict(data)
        provider = get_provider(data.get("provider"))
        caps = provider.capabilities()

        original = None
        if req.input_image:
            base = os.path.basename(req.input_image).rsplit(".", 1)[0]
            original = storage.original_path(base) or storage.resolve_image(req.input_image)

        context = PipelineContext(original_image=original)
        if original:
            context.faces = pipeline.detector.detect(original)
            if context.faces:
                context.preserved_face_indices = [f.index for f in context.faces]
        builder = builder_for(PROVIDER_DIALECTS.get(provider.name, "generic"))
        builder.build(req, context, caps)
        # Mirror the pipeline's own condition so the preview is truthful:
        # a genuine identity reference requires support, an original, and a
        # detected face with preservation requested.
        uses_identity_reference = bool(
            req.identity_requested()
            and caps.supports_face_preservation
            and original
            and context.faces
        )
        return jsonify(
            {
                "provider": provider.name,
                "dialect": PROVIDER_DIALECTS.get(provider.name, "generic"),
                "effective_prompt": context.effective_prompt,
                "clauses": {
                    "user": req.prompt,
                    "style": context.style_prompt,
                    "identity": context.identity_prompt,
                    "composition": context.composition_prompt,
                    "expression": context.expression_prompt,
                },
                "provider_params": context.provider_params,
                "uses_identity_reference": uses_identity_reference,
                "warnings": context.warnings,
            }
        )

    @app.route("/api/generate", methods=["POST"])
    def api_generate():
        data = request.get_json(silent=True) or {}
        req = GenerationRequest.from_dict(data)
        provider = get_provider(data.get("provider"))
        try:
            result = pipeline.execute(req, provider)
        except PipelineError as exc:
            return jsonify({"error": str(exc)}), 400
        payload = result.to_dict()
        payload["provider"] = provider.describe()
        payload["image_urls"] = [f"/media/{img.path}" for img in result.images]
        return jsonify(payload)

    @app.errorhandler(413)
    def too_large(_):
        return jsonify({"error": "The uploaded file is too large (limit 30 MB)."}), 413

    app.extensions["ais"] = {
        "storage": storage,
        "pipeline": pipeline,
        "config": config,
    }
    return app


def main() -> None:
    config = AppConfig.from_env()
    app = create_app(config)
    print(f"AI Image Studio running at http://{config.host}:{config.port}")
    print(f"Output directory: {config.output_dir}")
    app.run(host=config.host, port=config.port, debug=config.debug)


if __name__ == "__main__":
    main()