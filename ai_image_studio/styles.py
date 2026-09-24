"""Style catalogue and composition presets.

Each style carries a provider-neutral ``description`` plus a hint about how
compatible it is with identity preservation.  Highly stylised transforms
(cartoon/anime) naturally alter facial appearance, so the UI surfaces that as
``identity_fidelity`` rather than silently promising pixel-level sameness.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class StylePreset:
    key: str
    label: str
    description: str
    #: "high" | "moderate" | "low" -- how well identity survives this style.
    identity_fidelity: str
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


STYLES: dict[str, StylePreset] = {
    "none": StylePreset("none", "None", "", "high", ("neutral",)),
    "oil_painting": StylePreset(
        "oil_painting", "Oil Painting",
        "The painterly quality of traditional oil painting, visible brushwork "
        "and blended pigments.",
        "high", ("painting", "traditional"),
    ),
    "oil_painting_realism": StylePreset(
        "oil_painting_realism", "Oil Painting Realism",
        "A photorealistic oil painting with realistic lighting, accurate "
        "proportions and subtle brush texture. The subject remains lifelike.",
        "high", ("painting", "realism"),
    ),
    "watercolor": StylePreset(
        "watercolor", "Watercolor",
        "Soft watercolour washes on textured paper, translucent pigment and "
        "flowing edges.",
        "moderate", ("painting",),
    ),
    "pencil_sketch": StylePreset(
        "pencil_sketch", "Pencil Sketch",
        "A detailed graphite pencil sketch with fine hatching and shading.",
        "moderate", ("drawing",),
    ),
    "charcoal": StylePreset(
        "charcoal", "Charcoal",
        "Expressive charcoal drawing with deep blacks and smudged tonal "
        "transitions.",
        "moderate", ("drawing",),
    ),
    "vintage": StylePreset(
        "vintage", "Vintage",
        "A vintage film photograph with faded tones, soft contrast and subtle "
        "grain.",
        "high", ("photographic",),
    ),
    "renaissance": StylePreset(
        "renaissance", "Renaissance",
        "Renaissance portraiture: chiaroscuro lighting, rich earth tones and "
        "classical drapery.",
        "moderate", ("painting", "historical"),
    ),
    "cinematic": StylePreset(
        "cinematic", "Cinematic",
        "Cinematic colour grading, dramatic lighting and shallow depth of "
        "field.",
        "high", ("photographic", "film"),
    ),
    "black_and_white": StylePreset(
        "black_and_white", "Black & White",
        "A monochrome photograph with rich tonal range.",
        "high", ("photographic",),
    ),
    "digital_art": StylePreset(
        "digital_art", "Digital Art",
        "Polished digital illustration with clean rendering and vivid colour.",
        "moderate", ("digital",),
    ),
    "anime": StylePreset(
        "anime", "Anime",
        "Anime / manga illustration style with cel shading and stylised "
        "features.",
        "low", ("illustration", "stylised"),
    ),
    "cartoon": StylePreset(
        "cartoon", "Cartoon",
        "A bold cartoon illustration with simplified shapes and exaggerated "
        "features.",
        "low", ("illustration", "stylised"),
    ),
    "fantasy": StylePreset(
        "fantasy", "Fantasy",
        "A richly detailed fantasy painting with atmospheric lighting and "
        "imaginative detail.",
        "moderate", ("painting", "stylised"),
    ),
    "concept_art": StylePreset(
        "concept_art", "Concept Art",
        "Professional concept art with dramatic composition and painterly "
        "rendering.",
        "moderate", ("digital", "painting"),
    ),
}

DEFAULT_STYLE = "oil_painting_realism"


@dataclass(frozen=True)
class CompositionPreset:
    key: str
    label: str
    description: str

    def to_dict(self) -> dict:
        return asdict(self)


COMPOSITIONS: dict[str, CompositionPreset] = {
    "original": CompositionPreset("original", "Original", "Keep the original framing."),
    "portrait": CompositionPreset("portrait", "Portrait", "Head-and-shoulders portrait framing."),
    "close_up": CompositionPreset("close_up", "Close-up", "Tight close-up on the face."),
    "half_body": CompositionPreset("half_body", "Half Body", "Waist-up framing."),
    "full_body": CompositionPreset("full_body", "Full Body", "Full-length framing."),
    "wide": CompositionPreset("wide", "Wide Shot", "Wide environmental framing."),
}

ASPECT_RATIOS = ["original", "1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16"]


def style_text(style_key: str) -> str:
    preset = STYLES.get(style_key)
    if not preset or not preset.description:
        return ""
    return preset.description


def identity_fidelity(style_key: str) -> str:
    preset = STYLES.get(style_key)
    return preset.identity_fidelity if preset else "high"


def style_choices() -> list[dict]:
    return [s.to_dict() for s in STYLES.values()]


def composition_choices() -> list[dict]:
    return [c.to_dict() for c in COMPOSITIONS.values()]