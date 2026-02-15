# image_gen.py
# Keepsake Drift — image generation + local caching
#
# Generates atmospheric portrait images for each temporality based on current
# drift text. Images cached locally in data/images/ and regenerated every N ticks.
#
# Default: gpt-image-1-mini (low quality, 1024x1536) — ~$0.005-0.01/image
# Override via env: IMAGE_MODEL=dall-e-3  IMAGE_SIZE=1024x1792  IMAGE_QUALITY=standard

from __future__ import annotations

import base64
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from openai import OpenAI

log = logging.getLogger("image_gen")

# ---------------------
# Configuration
# ---------------------

IMAGE_DIR = Path(__file__).resolve().parent / "data" / "images"
IMAGE_GEN_INTERVAL = int(os.getenv("IMAGE_GEN_INTERVAL", "1"))
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-1-mini")
# gpt-image-1 family: 1024x1024, 1024x1536, 1536x1024
# dall-e-3: also supports 1024x1792, 1792x1024
IMAGE_SIZE = os.getenv("IMAGE_SIZE", "1024x1536")   # portrait ~2:3
IMAGE_QUALITY = os.getenv("IMAGE_QUALITY", "low")    # low (~$0.005) / medium (~$0.02) / high

# Subtle scene variation per temporality — what shifts within the base style
LENS_VISUAL: Dict[str, str] = {
    "human": "Intimate scale — a window, a doorway, a street corner. The warmth is close.",
    "liminal": "Threshold spaces — corridors, underpasses, edges between inside and outside.",
    "environment": "Weather and horizon visible — rain, mist, sky gradients at the edge of a city.",
    "digital": "Neon signs, screen reflections, LED glow bleeding into wet surfaces.",
    "infrastructure": "Roads, bridges, power lines, rail tracks — systemic forms receding into distance.",
    "more_than_human": "Trees, water, stone surfaces alongside urban edges — nature pressing through.",
}


def _truncate_drift(text: str, max_words: int = 180) -> str:
    """Truncate drift text to keep the image prompt within token limits."""
    words = (text or "").split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def _drift_abstraction_level(version: int) -> dict:
    """
    Returns stage metadata. The visual style is always the same —
    nighttime urban photography with shallow depth of field.
    The version only labels the entry for continuity reference.
    """
    return {
        "stage": f"drift_{version}",
        "style": "",  # style is defined in the base prompt, not per-stage
    }


def _build_prev_visual_description(mind_key: str, prev_version: int, prev_drift_text: str) -> str:
    """
    Construct a concise description of what the previous image depicted,
    so the new image can evolve from it visually.
    """
    if prev_version < 1 or not prev_drift_text:
        return ""
    prev_abs = _drift_abstraction_level(prev_version)
    excerpt = _truncate_drift(prev_drift_text, max_words=60)
    return (
        f"The previous image (drift {prev_version}, {prev_abs['stage']} stage) "
        f"depicted: {excerpt}"
    )


def _format_anchor(anchor) -> str:
    """
    Format a single anchor for the image prompt.
    Accepts either a plain string or a dict with 'phrase' and 'category' keys.
    Tags by category so the prompt can give category-specific guidance.
    """
    if isinstance(anchor, dict):
        phrase = anchor.get("phrase", "")
        cat = anchor.get("category", "sensory")
        return f"  [{cat}] {phrase}"
    return f"  [sensory] {anchor}"


def _build_image_prompt(
    *,
    mind_key: str,
    drift_text: str,
    drift_direction: str,
    perspective: str,
    version: int = 1,
    prev_visual_description: str = "",
    sensory_anchors: Optional[list] = None,
    infiltrating_imagery: Optional[List[str]] = None,
) -> str:
    """
    Build an image prompt from drift state.

    Two layers compose each image:
    1. SCENE (primary) — present-moment subjects determine WHAT the camera
       is pointed at. These change every drift, creating visual variety.
    2. EMOTIONAL DNA — invariable anchors shape HOW the scene feels:
       temperature, weight, light quality, mood. They do NOT dictate
       what literal objects appear. A 'wet earth' anchor means dampness
       and intimacy, not necessarily dirt.

    The result: each drift shows a DIFFERENT scene that carries the SAME
    emotional signature — like the same feeling recurring across different
    encounters.
    """
    lens_visual = LENS_VISUAL.get(mind_key, LENS_VISUAL["human"])
    drift_excerpt = _truncate_drift(drift_text, max_words=200)

    # Format anchors with category tags — cap at 8 to prevent dominance
    anchor_lines = []
    if sensory_anchors:
        for a in sensory_anchors[:8]:
            anchor_lines.append(_format_anchor(a))

    # Present-moment imagery (concrete subjects from current events)
    imagery_lines = []
    if infiltrating_imagery:
        for img in infiltrating_imagery[:6]:
            imagery_lines.append(f"  - {img}")

    parts = [
        "VISUAL STYLE (mandatory):",
        "A softly focused photograph with gentle, dreamlike clarity — NOT sharp, but NOT grainy or noisy. "
        "Clean image with smooth gradations and no visible grain, noise, or texture artifacts. "
        "The scene has a subtle softness, as if slightly out of focus or shot with a shallow depth of field. "
        "Edges are present but gentle, not razor-sharp. "
        "Color palette: warm amber, deep teal, muted earth tones. Rich dark shadows with soft transitions. "
        "NO grain, NO noise, NO film texture, NO digital artifacts. "
        "The mood is quiet, melancholic, intimate — like a fading memory that retains its emotional weight "
        "but has lost its crispness. A photograph that feels more like remembering than seeing.",
        "",
    ]

    # SCENE FIRST — present-moment subjects are the PRIMARY visual content
    if imagery_lines or drift_direction:
        parts.extend([
            "SCENE (this determines WHAT the image shows):",
            "The present moment determines the visible content. These subjects are WHAT the "
            "camera is pointed at — the actual objects, landscapes, textures in the photograph. "
            "Build the scene from these present-moment elements first. "
            "Each drift should show a DIFFERENT place, angle, or moment than last time.",
        ])
        if imagery_lines:
            parts.extend([
                "Present-moment subjects (render these as the primary visible content):",
                *imagery_lines,
            ])
        if drift_direction:
            parts.extend([
                f"Direction of present-moment drift: {drift_direction[:300]}",
            ])
        parts.append("")

    # EMOTIONAL DNA — anchors shape atmosphere, NOT literal objects
    if anchor_lines:
        parts.extend([
            "EMOTIONAL DNA (these infuse the scene with mood and feeling — NEVER render them literally):",
            "These fragments are the EMOTIONAL ESSENCE of a recurring memory. They shape the image's "
            "atmosphere, temperature, and psychological weight. CRITICAL: Do NOT depict these as literal objects. "
            "Instead, let them influence the scene's FEELING — the quality of light, the weight of shadows, "
            "the intimacy or distance of the framing, the emotional temperature of color and composition.",
            "",
            "How to translate anchors into FEELING (not literal objects):",
            "  [sensory] → atmospheric mood, tactile quality, emotional temperature",
            "    Example: 'wet earth' = heaviness, dampness, dark intimacy (manifest as: low light, "
            "    moisture in the air, ground-level perspective, muted colors — NOT literal soil)",
            "  [proper_noun] → spatial character, scale, light quality, regional feeling",
            "    Example: 'Nebraska City' = flatness, provincial scale, grain-colored warmth (manifest as: "
            "    horizontal composition, small-town quietness, amber-gold light — NOT the actual city)",
            "  [temporal] → light character, time-consciousness, transitional mood",
            "    Example: 'golden hour' = warm transitional glow, day-ending melancholy (manifest as: "
            "    amber-copper tones, long shadows, liminal feeling — NOT a literal sunset)",
            "",
            "Apply these emotional signatures SUBTLY to whatever scene the present-moment imagery builds. "
            "Let the anchors shape the atmosphere WITHOUT overpowering or contradicting the RSS-derived subjects. "
            "The present moment provides WHAT is shown; the anchors provide HOW it feels.",
            "",
            "Anchors to infuse (as feeling, not literal depiction):",
            *anchor_lines,
            "",
        ])

    parts.extend([
        f"MEMORY CONTEXT (narrative shape — not a scene description):",
        f"{drift_excerpt}",
        "",
        f"Scene framing: {lens_visual}",
        "",
        "Portrait orientation (2:3). No text, no words, no letters, no writing.",
        "",
        "GUARDRAILS:",
        "- No recognizable human faces (blur or silhouette only).",
        "- No culturally or religiously offensive imagery.",
        "- No religious symbols, places of worship, or sacred iconography.",
        "- No political imagery, flags, or military content.",
        "- No violence, weapons, or disturbing content.",
        "- No readable text, logos, or brand marks.",
    ])

    return "\n".join(parts)


def _image_path(mind_key: str, version: int) -> Path:
    """Return the local file path for a cached image."""
    return IMAGE_DIR / f"{mind_key}_v{version}.png"


def should_generate(version: int) -> bool:
    """Check if this version should trigger image generation."""
    interval = IMAGE_GEN_INTERVAL
    if interval <= 0:
        return False
    return version > 0 and (version % interval == 0)


def generate_drift_image(
    *,
    mind_key: str,
    version: int,
    drift_text: str,
    drift_direction: str = "",
    perspective: str = "",
    prev_drift_text: str = "",
    sensory_anchors: Optional[list] = None,
    infiltrating_imagery: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Generate an image for a temporality's current drift state.

    Default: gpt-image-1-mini (low, 1024x1536) — ~$0.005-0.01/image.
    Higher versions produce progressively more abstract/hallucinatory images.

    If prev_drift_text is provided, the image prompt describes the previous
    visual state so the new image EVOLVES from it (visual continuity).

    sensory_anchors: invariable sensory phrases that should resonate through
    every drift image, grounding the visual atmosphere in the original memory.

    infiltrating_imagery: concrete present-moment phrases (from Stage 1 lens
    interpretation) that should appear as VISIBLE SUBJECTS in the image —
    creating visual reminiscence that prevents the same scene repeating.

    Returns a tuple (local_file_path, prompt_text) if successful,
    or (None, "") on failure. When cached, returns (path, "") since
    the prompt is not stored alongside the cached file.
    """
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    out_path = _image_path(mind_key, version)

    # Already cached — skip (prompt not available for cached images)
    if out_path.exists() and out_path.stat().st_size > 0:
        log.info("Image already cached: %s", out_path)
        return str(out_path), ""

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        log.warning("No OPENAI_API_KEY — skipping image generation for %s v%d", mind_key, version)
        return None, ""

    # Build previous visual state for evolution continuity
    prev_visual = ""
    if version > 1 and prev_drift_text:
        prev_visual = _build_prev_visual_description(mind_key, version - 1, prev_drift_text)

    prompt = _build_image_prompt(
        mind_key=mind_key,
        drift_text=drift_text,
        drift_direction=drift_direction,
        perspective=perspective,
        version=version,
        prev_visual_description=prev_visual,
        sensory_anchors=sensory_anchors,
        infiltrating_imagery=infiltrating_imagery,
    )

    log.info("Generating image for %s v%d (model=%s, size=%s)", mind_key, version, IMAGE_MODEL, IMAGE_SIZE)

    try:
        client = OpenAI(timeout=90.0)

        if IMAGE_MODEL == "dall-e-3":
            # DALL-E 3: use URL response, then download
            resp = client.images.generate(
                model=IMAGE_MODEL,
                prompt=prompt,
                size=IMAGE_SIZE,
                quality=IMAGE_QUALITY,
                style="natural",
                response_format="url",
                n=1,
            )

            image_url = resp.data[0].url if resp.data else None
            if not image_url:
                log.error("DALL-E 3 returned no URL for %s v%d", mind_key, version)
                return None, prompt

            # Download image from temporary URL (expires in ~60 min)
            img_resp = requests.get(image_url, timeout=60)
            img_resp.raise_for_status()
            out_path.write_bytes(img_resp.content)

        else:
            # gpt-image-1 or other models: use b64_json
            resp = client.images.generate(
                model=IMAGE_MODEL,
                prompt=prompt,
                size=IMAGE_SIZE,
                quality=IMAGE_QUALITY,
                n=1,
            )

            b64_data = resp.data[0].b64_json if resp.data else None
            if not b64_data:
                log.error("Image model returned no b64 data for %s v%d", mind_key, version)
                return None, prompt

            out_path.write_bytes(base64.b64decode(b64_data))

        file_size_kb = out_path.stat().st_size / 1024
        log.info("Saved image: %s (%.1f KB)", out_path, file_size_kb)

        # Log revised prompt if available (DALL-E 3 revises prompts)
        revised = getattr(resp.data[0], "revised_prompt", None) if resp.data else None
        if revised:
            log.debug("Revised prompt for %s v%d: %s", mind_key, version, revised[:200])

        return str(out_path), prompt

    except Exception as e:
        log.error("Image generation failed for %s v%d: %s", mind_key, version, e)
        # Clean up partial file
        if out_path.exists():
            try:
                out_path.unlink()
            except Exception:
                pass
        return None, prompt


def generate_all_drift_images(
    minds: Dict[str, dict],
) -> Dict[str, Optional[str]]:
    """
    Generate images for multiple temporalities in one call.
    minds: {mind_key: {"version": int, "drift_text": str, "drift_direction": str, "perspective": str}}
    Returns: {mind_key: local_path_or_None}
    """
    results: Dict[str, Optional[str]] = {}
    for mind_key, info in minds.items():
        version = int(info.get("version", 0))
        if not should_generate(version):
            results[mind_key] = None
            continue
        path, _prompt = generate_drift_image(
            mind_key=mind_key,
            version=version,
            drift_text=info.get("drift_text", ""),
            drift_direction=info.get("drift_direction", ""),
            perspective=info.get("perspective", ""),
        )
        results[mind_key] = path
    return results


def find_latest_image(mind_key: str, up_to_version: int) -> Optional[str]:
    """
    Find the most recent generated image for a mind, looking back from up_to_version.
    Returns the local file path or None.
    """
    if not IMAGE_DIR.exists():
        return None

    # Search backwards from up_to_version
    interval = max(1, IMAGE_GEN_INTERVAL)
    # Start from the highest version that would have generated (round down to interval)
    start = (up_to_version // interval) * interval

    for v in range(start, 0, -interval):
        path = _image_path(mind_key, v)
        if path.exists() and path.stat().st_size > 0:
            return str(path)

    return None


def find_image_history(mind_key: str, up_to_version: int, count: int = 3) -> list:
    """
    Find the N most recent distinct images for a mind, looking back from up_to_version.
    Returns a list of local file paths, newest first. Length 0..count.
    Used for temporal layering (n, n-1, n-2 blended in the frontend).
    """
    if not IMAGE_DIR.exists():
        return []

    results = []
    interval = max(1, IMAGE_GEN_INTERVAL)
    start = (up_to_version // interval) * interval

    for v in range(start, 0, -interval):
        path = _image_path(mind_key, v)
        if path.exists() and path.stat().st_size > 0:
            results.append(str(path))
            if len(results) >= count:
                break

    return results
