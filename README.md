# AI Image Studio

An image generation and editing application built around one requirement:
**when you edit a photograph of a person, the person should still look like
the same person.**

The architecture keeps three concerns separate so they can be controlled
independently:

```
IDENTITY   what makes the person recognisable   (preserve when requested)
CONTENT    pose, framing, composition, layout   (preserve when requested)
STYLE      the requested visual transformation  (always allowed to change)
```

---

## Quick start

### From source

```bash
pip install -r requirements.txt
python run.py
# open http://localhost:12000
```

Configuration (all optional):

| Variable | Default | Purpose |
| --- | --- | --- |
| `AIS_PORT` | `12000` | HTTP port |
| `AIS_HOST` | `0.0.0.0` | Bind address |
| `AIS_OUTPUT_DIR` | `./AIImageStudio` | Where originals and results are stored |
| `AIS_PROVIDER` | `local` | Default backend |
| `AIS_IDENTITY_CHECK` | `1` | Run the post-generation identity check |
| `AIS_OPEN_BROWSER` | `0` source / `1` exe | Open the GUI in a browser on start |
| `IMAGE_API_URL` / `IMAGE_API_KEY` | – | Endpoint and key for the remote backend |

### Windows executable

Download `ai_image_studio.exe` from the
[releases page](https://github.com/Askcodex-do/text2image/releases) and run it.
It opens the GUI in your browser and writes all output to an `AIImageStudio`
folder created beside the executable. Nothing is installed and no Python is
required.

The executable is produced by the
[build workflow](.github/workflows/build-exe.yml) on a native Windows runner.
It can also be cross-built on Linux with Wine:

```bash
bash packaging/build_exe.sh          # -> ~/.ai_image_studio-build/dist/ai_image_studio.exe
```

Run the tests:

```bash
python -m pytest tests/ -q
```

---

## What actually preserves identity

This is the most important thing to understand about this application.

### The `local` provider — genuine identity preservation

The bundled OpenCV provider transforms the image spatially, then **composites
the original face region back over the stylised result** through a feathered
mask. The person's real facial pixels survive the style change, so identity is
preserved as a matter of fact, not as a prompt request.

Verified on the test fixtures: at `Maximum` strength the mean absolute
difference between the generated and original face pixels is ~1.6 (out of 255),
versus ~34 with preservation off.

The `Face Preservation` strength controls two things at once:

| Strength | Face kept | Style strength |
| --- | --- | --- |
| Low | softly blended | strongest |
| Medium | more | strong |
| High | most | moderate |
| Maximum | almost untouched | gentlest |

Because the transform is spatial and local, `Preserve Composition` and
`Preserve Expression` are inherently satisfied for this provider.

### The `remote` provider — honest about its limits

The remote provider only claims identity preservation when the backend
actually reports that it supports it (via a `capabilities` object). When it
does, the **untouched original image is sent as an identity reference** —
never as merely a prompt instruction.

When the backend does *not* report support:

* `supports_face_preservation()` returns `False`
* the GUI **disables** the Preserve Face controls rather than pretending
* the strength selector collapses to `Off`
* the prompt builder records an explicit warning that preservation is
  prompt-only and not guaranteed

### What the application refuses to do

It never claims that a numerical similarity score proves two faces are the
same individual. The built-in check is a **non-biometric quality-control
signal** and is labelled as such everywhere it appears.

---

## Pipeline

```
Original Image
      │
      ▼
Image Validation
      │
      ▼
Face Detection ────────► (optional: no face → continue normally)
      │
      ▼
Face / Identity Reference
      │
      ▼
Prompt Builder
      │
 ┌────┴────┐
 │         │
User      Style
Prompt    Prompt
 │         │
 └────┬────┘
      ▼
Image Generation
      │
      ▼
Identity Checking ─────► (non-biometric signal)
      │
      ▼
Final Image
```

Key properties:

* Face detection is **not** required. A landscape or an image where detection
  fails proceeds normally with a recorded warning.
* The original image is carried through the whole request
  (`PipelineContext.original_image`) so any provider capable of true
  reference-image conditioning can use it.
* Multiple people are supported: all detected faces are preserved by default,
  and `selected_face_indices` lets the UI preserve a subset.

---

## Original files are immutable

Hard rule enforced in code and covered by tests:

```
photos/person.jpg                        ← your file, never written to
AIImageStudio/originals/person_ab12….png ← immutable working copy
AIImageStudio/images/person_oil_painting_realism_local_001.png
```

`Storage.is_original_path()` guards every cleanup path, and
`test_original_is_never_modified` asserts the user's source file is
byte-for-byte identical after generation.

---

## Provider abstraction

```python
class ImageProvider:
    def generate(self, request, context): ...
    def edit(self, request, context): ...

    def capabilities(self) -> ProviderCapabilities: ...
    def supports_face_preservation(self) -> bool:
        return self.capabilities().supports_face_preservation
```

`ProviderCapabilities` reports, per backend:

`supports_face_preservation`, `supports_reference_image`,
`supports_multiple_faces`, `supports_composition_control`,
`supports_expression_control`, `supports_identity_check`,
`supports_negative_prompt`, `supports_seed`, `is_remote`,
`max_images_per_request`, `strength_mapping`, `notes`.

The GUI reads `/api/config` and disables every control the selected backend
cannot honour.

## Prompt building is provider-specific

There is no universal prompt. Each provider is mapped to a dialect and only
receives clauses it understands:

* `local` → `stable_diffusion` (comma-separated tags)
* `remote` → `openai` (prose)
* fallback → `generic`

Identity, composition and expression are assembled as **separate clauses** so
that a provider with real reference-image conditioning relies on the image,
not on the text. `/api/preview-prompt` shows exactly what will be sent.

An explicit user instruction always wins: a prompt such as *"make the person
smile naturally"* suppresses expression preservation, and the UI says so.

---

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | GUI |
| `GET` | `/api/config` | Provider capabilities, styles, options |
| `POST` | `/api/upload` | Store an immutable original, detect faces |
| `POST` | `/api/faces` | Re-run detection for an image id |
| `POST` | `/api/preview-prompt` | Show the exact provider-bound prompt |
| `POST` | `/api/generate` | Generate / edit |
| `GET` | `/media/<path>` | Serve originals and results |

Example:

```bash
curl -F image=@person.jpg http://localhost:12000/api/upload
curl -X POST http://localhost:12000/api/generate -H 'Content-Type: application/json' -d '{
  "prompt": "Convert this photograph into a realistic oil painting.",
  "input_image": "person_ab12cd34ef56",
  "style": "oil_painting_realism",
  "face_preservation_strength": "high",
  "preserve_face": true,
  "preserve_composition": true,
  "preserve_expression": true,
  "number_of_images": 3
}'
```

The response includes the effective prompt, the detected faces, per-image
identity-check results, and the untouched `original_path`.

---

## Styles and identity fidelity

Highly stylised transforms (Anime, Cartoon) naturally alter facial appearance.
The style catalogue records an `identity_fidelity` hint, and the UI describes
preservation as an *attempt*, never a pixel-level guarantee:

* `high` — Oil Painting Realism, Vintage, Cinematic, Black & White
* `moderate` — Watercolour, Sketch, Charcoal, Renaissance, Digital, Fantasy,
  Concept Art
* `low` — Anime, Cartoon

---

## Project layout

```
ai_image_studio/
  models.py               request / capability / context models
  styles.py               style + composition catalogue
  config.py               environment configuration
  app.py                  Flask app factory and HTTP API
  providers/
    base.py               ImageProvider abstraction
    local_provider.py     OpenCV provider with real face compositing
    remote_provider.py    capability-aware HTTP client
    registry.py           provider registry
  services/
    face_service.py       detection (Haar + NMS) and identity verification
    prompt_builder.py     provider-specific prompt dialects
    pipeline.py           face-aware editing pipeline
    storage.py            immutable-original storage
  templates/ static/      GUI
packaging/
  ai_image_studio.spec    PyInstaller configuration
  build_exe.sh            Wine-based cross build for Linux hosts
.github/workflows/
  build-exe.yml           native Windows build, smoke test and release upload
tests/                    49 tests covering pipeline, API, packaging and
                          honesty rules
```

## Test assets

`tests/assets/` contains small face crops derived from OpenCV's standard
sample images, which are Apache-2.0 licensed. Haar cascades detect real
photographic faces far more reliably than synthetic drawings, so using real
crops exercises the true detection path.

## Limitations

* The default detector detects frontal faces. Profile faces may not be found;
  the pipeline then continues without preservation and says so.
* The identity check is a heuristic (normalised cross-correlation plus skin
  tone), not a recognition model. It is a review signal only.
* The `local` provider composites the original face region, so for extreme
  styles (Anime, Cartoon) the preserved face will look photographic against a
  stylised remainder. This is a deliberate, honest trade-off favouring
  identity over style uniformity.