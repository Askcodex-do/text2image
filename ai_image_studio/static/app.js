/* AI Image Studio front-end.
 * Drives the GUI from /api/config capabilities so controls only appear
 * meaningful when the selected backend genuinely supports them.
 */
(function () {
  "use strict";

  const state = {
    config: null,
    providers: {},
    uploaded: null,
    faces: [],
  };

  const $ = (id) => document.getElementById(id);
  const el = {
    provider: $("provider"),
    uploadBtn: $("upload-btn"),
    fileInput: $("file-input"),
    prompt: $("prompt"),
    style: $("style"),
    composition: $("composition"),
    aspectRatio: $("aspect-ratio"),
    faceStrength: $("face-strength"),
    numberOfImages: $("number-of-images"),
    preserveFace: $("preserve-face"),
    preserveComposition: $("preserve-composition"),
    preserveExpression: $("preserve-expression"),
    generateBtn: $("generate-btn"),
    previewBtn: $("preview-btn"),
    promptPreview: $("prompt-preview"),
    gallery: $("gallery"),
    error: $("error"),
    uploadStatus: $("upload-status"),
    facePanel: $("face-panel"),
    faceCount: $("face-count"),
    faceList: $("face-list"),
    detectorState: $("detector-state"),
    capabilityNote: $("capability-note"),
  };

  function showError(message) {
    el.error.textContent = message;
    el.error.hidden = !message;
  }

  function currentProvider() {
    return state.providers[el.provider.value] || null;
  }

  /* Apply provider capabilities to the UI: disable controls that can have no
   * effect and explain why, rather than pretending they work. */
  function applyCapabilities() {
    const provider = currentProvider();
    if (!provider) return;
    const caps = provider.capabilities;

    const faceSupported = caps.supports_face_preservation;
    el.preserveFace.disabled = !faceSupported;
    el.faceStrength.disabled = !faceSupported;
    el.preserveExpression.disabled = !caps.supports_expression_control;
    el.preserveComposition.disabled = !caps.supports_composition_control;

    const strengths = caps.supported_strengths || ["off"];
    Array.from(el.faceStrength.options).forEach((option) => {
      option.disabled = !strengths.includes(option.value);
    });
    if (!strengths.includes(el.faceStrength.value) && strengths.length) {
      el.faceStrength.value = strengths[strengths.length - 1];
    }

    const maxImages = caps.max_images_per_request || 4;
    Array.from(el.numberOfImages.options).forEach((option) => {
      option.disabled = parseInt(option.value, 10) > maxImages;
    });
    if (parseInt(el.numberOfImages.value, 10) > maxImages) {
      el.numberOfImages.value = String(maxImages);
    }

    const notes = [];
    if (!provider.available) {
      notes.push(provider.unavailable_reason || "This provider is not configured.");
    }
    if (!faceSupported) {
      notes.push(
        "This provider does not support identity preservation. The Preserve " +
        "Face setting is shown for completeness but has no guaranteed effect."
      );
    } else {
      notes.push(
        "This provider supports identity preservation using the original " +
        "image as a reference. Results vary by image and style."
      );
    }
    (caps.notes || []).forEach((note) => notes.push(note));
    el.capabilityNote.textContent = notes.join(" ");
    el.capabilityNote.classList.toggle("ok", faceSupported && provider.available);
  }

  async function loadConfig() {
    const response = await fetch("/api/config");
    state.config = await response.json();
    (state.config.providers || []).forEach((p) => {
      state.providers[p.name] = p;
    });
    applyCapabilities();
  }

  function renderFaces(data) {
    state.faces = data.faces || [];
    el.facePanel.hidden = state.faces.length === 0;
    el.faceCount.textContent = `Faces detected: ${state.faces.length}`;
    el.detectorState.textContent = data.detector_available === false
      ? "detector unavailable"
      : "";
    el.faceList.innerHTML = "";
    state.faces.forEach((face, i) => {
      const label = document.createElement("label");
      label.className = "face-chip active";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = true;
      checkbox.dataset.index = face.index;
      checkbox.addEventListener("change", () => {
        label.classList.toggle("active", checkbox.checked);
      });
      const span = document.createElement("span");
      span.textContent = `Face ${i + 1}`;
      label.append(checkbox, span);
      el.faceList.append(label);
    });
  }

  function selectedFaceIndices() {
    return el.faceList.querySelectorAll('input[type="checkbox"]');
  }

  async function handleUpload(file) {
    const form = new FormData();
    form.append("image", file);
    el.uploadBtn.textContent = "…";
    try {
      const response = await fetch("/api/upload", { method: "POST", body: form });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Upload failed.");
      state.uploaded = data.image;
      el.uploadStatus.hidden = false;
      el.uploadStatus.textContent =
        `${data.image.filename} — ${data.image.width}×${data.image.height} px. ${data.message}`;
      el.uploadStatus.classList.toggle("warn", data.face_count === 0);
      renderFaces(data);
      showError("");
    } catch (error) {
      showError(error.message);
    } finally {
      el.uploadBtn.textContent = "+";
    }
  }

  function buildPayload() {
    const boxes = selectedFaceIndices();
    const selected = Array.from(boxes)
      .filter((box) => box.checked)
      .map((box) => parseInt(box.dataset.index, 10));
    const useSelection = boxes.length > 0 && selected.length !== boxes.length;

    return {
      prompt: el.prompt.value.trim(),
      input_image: state.uploaded ? state.uploaded.id : null,
      style: el.style.value,
      composition: el.composition.value,
      aspect_ratio: el.aspectRatio.value,
      preserve_face: el.preserveFace.checked && !el.preserveFace.disabled,
      face_preservation_strength: el.faceStrength.value,
      preserve_composition: el.preserveComposition.checked && !el.preserveComposition.disabled,
      preserve_expression: el.preserveExpression.checked && !el.preserveExpression.disabled,
      number_of_images: parseInt(el.numberOfImages.value, 10),
      provider: el.provider.value,
      selected_face_indices: useSelection ? selected : null,
    };
  }

  function identityBadge(check) {
    if (!check || !check.performed) return null;
    const badge = document.createElement("span");
    const ok = check.verdict === "completed";
    badge.className = "badge " + (ok ? "ok" : "warn");
    badge.textContent = ok
      ? `Identity check completed (${check.score})`
      : `Identity: review recommended (${check.score})`;
    badge.title =
      "Non-biometric quality-control signal, not proof of identity. " +
      (check.message || "");
    return badge;
  }

  function renderImages(payload) {
    el.gallery.innerHTML = "";
    payload.images.forEach((image, i) => {
      const card = document.createElement("div");
      card.className = "card";

      const img = document.createElement("img");
      img.src = `/media/${image.path}`;
      img.alt = `Generated image ${i + 1}`;
      img.loading = "lazy";

      const body = document.createElement("div");
      body.className = "card-body";
      const path = document.createElement("div");
      path.className = "path";
      path.textContent = image.path;
      body.append(path);

      const badge = identityBadge(image.identity_check);
      if (badge) body.append(badge);
      else {
        const none = document.createElement("span");
        none.className = "badge off";
        none.textContent = "Identity check not performed";
        body.append(none);
      }
      if (image.original_path) {
        const original = document.createElement("div");
        original.className = "path";
        original.textContent = `original: ${image.original_path}`;
        body.append(original);
      }

      card.append(img, body);
      el.gallery.append(card);
    });
  }

  async function generate() {
    showError("");
    el.promptPreview.hidden = true;
    const payload = buildPayload();
    if (!payload.prompt) {
      showError("Please describe what you want to create or change.");
      return;
    }
    el.generateBtn.disabled = true;
    el.generateBtn.textContent = "Generating…";
    try {
      const response = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Generation failed.");
      renderImages(data);
      (data.pipeline.warnings || []).forEach((warning) => showError(warning));
    } catch (error) {
      showError(error.message);
    } finally {
      el.generateBtn.disabled = false;
      el.generateBtn.textContent = "Generate";
    }
  }

  async function previewPrompt() {
    showError("");
    const payload = buildPayload();
    try {
      const response = await fetch("/api/preview-prompt", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Preview failed.");
      const lines = [
        `Provider: ${data.provider}  (prompt dialect: ${data.dialect})`,
        `Identity reference used: ${data.uses_identity_reference ? "yes" : "no"}`,
        "",
        "Effective prompt:",
        data.effective_prompt || "(empty)",
        "",
        "Provider params: " + JSON.stringify(data.provider_params),
      ];
      (data.warnings || []).forEach((w) => lines.push("Warning: " + w));
      el.promptPreview.textContent = lines.join("\n");
      el.promptPreview.hidden = false;
    } catch (error) {
      showError(error.message);
    }
  }

  el.uploadBtn.addEventListener("click", () => el.fileInput.click());
  el.fileInput.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (file) handleUpload(file);
    el.fileInput.value = "";
  });
  el.provider.addEventListener("change", applyCapabilities);
  el.generateBtn.addEventListener("click", generate);
  el.previewBtn.addEventListener("click", previewPrompt);

  loadConfig().catch((error) => showError(error.message));
})();