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

import time as _time

import requests
from openai import OpenAI

try:
    import llm_logger
except Exception:
    llm_logger = None  # type: ignore

try:
    from PIL import Image as _PILImage
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

log = logging.getLogger("image_gen")

# ── Post-generation quality gate ──
MAX_REGEN_ATTEMPTS = 5  # retry up to 3x if quality/content check fails

def _image_quality_ok(path: Path) -> bool:
    """
    Basic quality gate: reject images that are mostly blown-out (white),
    mostly black, or mostly transparent. Returns True if image passes.
    Checks: average brightness, top-half vs bottom-half balance,
    percentage of near-white / near-black pixels, and transparency.
    """
    if not _HAS_PIL:
        return True  # can't check without PIL — accept

    try:
        raw = _PILImage.open(path)

        # ── Transparency check: reject if >30% of pixels are transparent ──
        if raw.mode in ("RGBA", "LA", "PA"):
            alpha = raw.split()[-1]
            alpha_data = list(alpha.getdata())
            transparent_pct = sum(1 for a in alpha_data if a < 30) / len(alpha_data) if alpha_data else 0
            if transparent_pct > 0.05:
                log.warning("Quality gate: %.0f%% transparent pixels — rejecting", transparent_pct * 100)
                return False

        # Composite on white background before grayscale conversion
        # so transparent areas are evaluated as white (how browsers render them)
        if raw.mode == "RGBA":
            bg = _PILImage.new("RGB", raw.size, (255, 255, 255))
            bg.paste(raw, mask=raw.split()[3])
            img = bg.convert("L")
        else:
            img = raw.convert("L")
        _get_data = getattr(img, "get_flattened_data", None) or img.getdata
        pixels = list(_get_data())
        total = len(pixels)
        if total == 0:
            return False

        avg = sum(pixels) / total
        near_white = sum(1 for p in pixels if p > 240) / total
        near_black = sum(1 for p in pixels if p < 15) / total

        # Check top third vs bottom third balance
        w, h = img.size
        _top = img.crop((0, 0, w, h // 3))
        _bot = img.crop((0, 2 * h // 3, w, h))
        _get_top = getattr(_top, "get_flattened_data", None) or _top.getdata
        _get_bot = getattr(_bot, "get_flattened_data", None) or _bot.getdata
        top_third = list(_get_top())
        bot_third = list(_get_bot())
        top_avg = sum(top_third) / len(top_third) if top_third else 128
        bot_avg = sum(bot_third) / len(bot_third) if bot_third else 128

        # Contrast check — standard deviation of pixel values
        mean = avg
        variance = sum((p - mean) ** 2 for p in pixels) / total
        std_dev = variance ** 0.5

        # Washed-out check — pixels in the 180-240 "gray-bright" band
        washed_out = sum(1 for p in pixels if 180 < p < 240) / total

        # Reject conditions — thresholds tuned for gpt-image-1 high quality,
        # which renders darker / more cinematic images than gpt-image-1-mini.
        if near_white > 0.40:
            log.warning("Quality gate: %.0f%% near-white pixels (avg=%.0f)", near_white * 100, avg)
            return False
        if near_black > 0.85:
            log.warning("Quality gate: %.0f%% near-black pixels (avg=%.0f)", near_black * 100, avg)
            return False
        if abs(top_avg - bot_avg) > 140:
            log.warning("Quality gate: extreme vertical imbalance (top=%.0f, bot=%.0f)", top_avg, bot_avg)
            return False
        if std_dev < 12:
            log.warning("Quality gate: very low contrast (std_dev=%.1f, avg=%.0f) — flat image", std_dev, avg)
            return False
        if washed_out > 0.50:
            log.warning("Quality gate: %.0f%% washed-out pixels (180-240 range, avg=%.0f)", washed_out * 100, avg)
            return False
        if avg > 220:
            log.warning("Quality gate: overall too bright (avg=%.0f)", avg)
            return False

        return True
    except Exception as e:
        log.warning("Quality gate error (accepting image): %s", e)
        return True


def _image_content_ok(path: Path) -> tuple:
    """
    Vision-based gate: reject images that contain living beings OR look
    corrupted / unfinished / excessively blown-out.
    Returns (True, "") if clean, (False, reason) if rejected.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return True, ""

    try:
        import base64 as _b64
        img_bytes = path.read_bytes()
        b64_img = _b64.b64encode(img_bytes).decode("utf-8")

        client = OpenAI(timeout=30.0)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=80,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "Check this image for TWO things:\n"
                        "1. LIVING BEINGS: Does it contain any people, human figures, silhouettes, "
                        "shadows of humans, ghost-like presences, faces (including on screens/phones), "
                        "hands, animals, birds, flamingos, fish, insects, or any living CREATURE "
                        "(a being that moves on its own)? FAIL if any are present — even faint or stylized ones. "
                        "PLANTS (flowers, grass, trees, leaves, moss, vines) are SCENERY and NOT living "
                        "beings for this check — do NOT fail for plants.\n"
                        "2. IMAGE QUALITY: A good image is a CLEAR photographic scene of a "
                        "recognisable PLACE (room, street, corridor, landscape, building). "
                        "FAIL the image if: it is mostly white or blown-out; mostly one flat "
                        "colour with no detail; looks unfinished, corrupted, or glitched; "
                        "is too abstract to read as a real location; has large featureless "
                        "areas (blank walls taking up most of the frame with no other detail); "
                        "or looks washed-out/gray with no depth or contrast.\n"
                        "Reply ONLY with: PASS (plants count as scenery, not living beings) "
                        "or FAIL: <what was found>"
                    )},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{b64_img}",
                        "detail": "low",
                    }},
                ],
            }],
        )
        answer = resp.choices[0].message.content.strip() if resp.choices else "PASS"
        if answer.upper().startswith("PASS"):
            log.info("Content gate PASSED for %s", path.name)
            return True, ""
        else:
            reason = answer.replace("FAIL:", "").strip()
            log.warning("Content gate FAILED for %s: %s", path.name, reason)
            return False, reason
    except Exception as e:
        log.warning("Content gate error (accepting image): %s", e)
        return True, ""


def _image_quality_reason(path: Path) -> str:
    """Return a human-readable reason for quality gate failure."""
    if not _HAS_PIL:
        return "PIL not available"
    try:
        img = _PILImage.open(path).convert("L")
        _get_data = getattr(img, "get_flattened_data", None) or img.getdata
        pixels = list(_get_data())
        total = len(pixels)
        if total == 0:
            return "empty image"
        near_white = sum(1 for p in pixels if p > 240) / total
        near_black = sum(1 for p in pixels if p < 15) / total
        w, h = img.size
        _top = img.crop((0, 0, w, h // 3))
        _bot = img.crop((0, 2 * h // 3, w, h))
        _get_top = getattr(_top, "get_flattened_data", None) or _top.getdata
        _get_bot = getattr(_bot, "get_flattened_data", None) or _bot.getdata
        top_third = list(_get_top())
        bot_third = list(_get_bot())
        top_avg = sum(top_third) / len(top_third) if top_third else 128
        bot_avg = sum(bot_third) / len(bot_third) if bot_third else 128
        reasons = []
        if near_white > 0.35:
            reasons.append(f"{near_white*100:.0f}% blown-out white")
        if near_black > 0.50:
            reasons.append(f"{near_black*100:.0f}% near-black")
        if abs(top_avg - bot_avg) > 140:
            reasons.append(f"vertical imbalance (top={top_avg:.0f}, bot={bot_avg:.0f})")
        return "; ".join(reasons) if reasons else "unknown"
    except Exception as e:
        return f"check error: {e}"


# ---------------------
# Configuration
# ---------------------

IMAGE_DIR = Path(__file__).resolve().parent / "data" / "images"
IMAGE_GEN_INTERVAL = int(os.getenv("IMAGE_GEN_INTERVAL", "1"))
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-1")
# gpt-image-1 family: 1024x1024, 1024x1536, 1536x1024
# dall-e-3: also supports 1024x1792, 1792x1024
IMAGE_SIZE = os.getenv("IMAGE_SIZE", "1024x1536")    # portrait ~2:3
IMAGE_QUALITY = os.getenv("IMAGE_QUALITY", "high")   # low (~$0.011) / medium (~$0.042) / high (~$0.25)

# Geolocation context per instance — KD_GEO_KEY overrides KD_SECOND_LANG
# (the EN instance keeps KD_SECOND_LANG unset and binds geographically via KD_GEO_KEY=en)
_LOCALE_GEO = {
    "ar":    "Dubai / Gulf region. Modern Middle-Eastern city — contemporary buildings, desert climate, warm light.",
    "en":    "New York City. Dense American metropolis — brick walk-ups, fire escapes, cast-iron facades, brownstones, bodegas, subway grates and steam, mixed daylight bouncing between high-rises, gritty street-level texture.",
    "el":    "Athens / Mediterranean city. White walls, warm stone, balconies, bright Mediterranean light.",
    "pt-br": "São Paulo. Dense urban metropolis — concrete high-rises, grey overpasses, narrow streets, mixed architecture from art deco to brutalist, warm overcast light, sprawling cityscape.",
}
_GEO_KEY = (os.getenv("KD_GEO_KEY", "").strip().lower()
            or os.getenv("KD_SECOND_LANG", "ar").strip().lower())
_INSTANCE_GEO = _LOCALE_GEO.get(_GEO_KEY, _LOCALE_GEO["ar"])

# Temporal lens qualities — HOW each lens sees, not WHAT it depicts.
# The memory's own emotional scene remains the subject; these lenses
# alter the *quality of attention* — what comes into focus, what softens,
# how light and atmosphere behave.  Cycles per version.
LENS_VISUAL: Dict[str, List[str]] = {
    "human": [
        "Intimate focus — as if remembering from very close. The periphery softens into emotional blur while details sharpen where feeling was strongest. Shallow depth of field.",
        "The quality of personal recollection — tender proximity, as if the image is held in the hand. Surfaces are vivid and present, lit by whatever light the memory holds.",
        "Memory at the threshold of sleep — the scene has the softness of something remembered just before waking. Familiar, gentle, slightly uncertain at the edges.",
    ],
    "liminal": [
        "Threshold quality — the scene exists between two states. Light and shadow mark a transition. Something is about to change but hasn't yet.",
        "The suspended moment — time has paused between one thing ending and another beginning. The air itself feels held.",
        "Passage quality — everything has the character of being seen while moving through. Not settling, not staying.",
    ],
    "environment": [
        "Expansive awareness — the scene breathes outward. Weather, air quality, the pressure of sky on surfaces. The atmosphere is a character.",
        "Ecological attention — surfaces register temperature, moisture, the slow work of climate. Materials show their relationship to elements.",
        "The long patience — time moves at geological scale. Human presence is implied by the spaces left behind, not by people.",
    ],
    "digital": [
        "Mediated quality — as if the memory was recalled through a screen. Clean digital clarity, the precision of something transmitted rather than touched.",
        "Signal quality — the scene has the feel of something received from a distance. A subtle latency, as if light arrived a moment late.",
        "Processed clarity — this memory has been stored and retrieved digitally. Slightly flattened depth, even illumination, precise edges.",
    ],
    "infrastructure": [
        "Structural awareness — the scene reveals underlying systems. The hidden geometry beneath the visible. Pipes, wires, frameworks sensed through walls.",
        "Systemic attention — the eye notices what usually goes unseen: the repetition of units, the logic of grids, the patterns that organize space.",
        "Maintenance perspective — the quality of seeing a space from behind, from below, from the service side. The scene is familiar but viewed from an unusual angle.",
    ],
    "more_than_human": [
        "Non-anthropocentric scale — perspective shifts beyond the human. The scene is seen from the viewpoint of what persists before and after people.",
        "Geological patience — deep time looking through the scene. Stone, water, root systems have their own slow attention.",
        "Ecological entanglement — the boundary between built and grown dissolves. Each space interpenetrates the other.",
    ],
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
    photorealistic photography with shallow depth of field.
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
    excerpt = _truncate_drift(prev_drift_text, max_words=25)
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


import re as _re


def _sanitize_excerpt(text: str) -> str:
    """Strip sentences that would cause DALL-E to render living beings or faces.

    Strategy: split into sentences, drop any sentence containing trigger words,
    rejoin.  This is aggressive but necessary — DALL-E infers visuals from
    contextual phrases (e.g. "pale pink bodies on thin legs" → flamingos)
    even when the noun itself is removed.
    """
    # Trigger words: anything that implies a living being or a visible face/body
    _triggers = _re.compile(
        r'(?:flamingo|flamingos|bird|birds|cat|cats|dog|dogs|fish|fishes|'
        r'animal|animals|creature|creatures|insect|insects|butterfly|butterflies|'
        r'pigeon|pigeons|crow|crows|sparrow|sparrows|eagle|eagles|falcon|falcons|'
        r'camel|camels|horse|horses|goat|goats|sheep|dolphin|dolphins|whale|whales|'
        r'turtle|turtles|lizard|lizards|snake|snakes|ant|ants|bee|bees|'
        r'stray|flock|herd|swarm|wildlife|fauna|'
        # body / face triggers
        r'face|faces|facial|video[\s-]?call|facetime|binocular|binoculars|'
        r'pale\s+pink|pink\s+bod|pink\s+form|standing\s+on\s+legs|'
        r'thin\s+sticks|wading|wings?\s+spread|feather|feathers|beak|beaks|'
        r'nest|nests|nesting|hatching|migration|migratory)',
        _re.IGNORECASE,
    )

    # Split on sentence boundaries (. ! ? or — followed by space + capital)
    sentences = _re.split(r'(?<=[.!?—])\s+', text)
    kept = [s for s in sentences if not _triggers.search(s)]

    cleaned = ' '.join(kept)
    # Collapse whitespace
    cleaned = _re.sub(r'  +', ' ', cleaned)
    return cleaned.strip()


# ── Scene-setting cache ──
# The physical setting of a memory barely changes between drift versions —
# it's always the same balcony, car park, creek, etc. Extracting it via
# GPT-4o-mini every tick wastes ~2-3s per image (6 images = 12-18s).
# We extract once and cache in originals.meta_json, then reuse every tick.
_scene_cache: Dict[str, str] = {}  # in-memory: mind_key → setting text


def _load_scene_cache_from_db() -> None:
    """Load cached scene settings from originals.meta_json at startup."""
    global _scene_cache
    db_path = os.getenv("KD_DB_PATH", "") or os.path.join(
        os.path.dirname(__file__), "data", "keepsake.sqlite"
    )
    if not os.path.exists(db_path):
        return
    try:
        import sqlite3, json as _json
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT temporality, meta_json FROM originals WHERE meta_json IS NOT NULL"
        ).fetchall()
        conn.close()
        for r in rows:
            try:
                meta = _json.loads(r["meta_json"] or "{}")
                cached = meta.get("scene_setting", "")
                if cached and len(cached) > 20:
                    _scene_cache[r["temporality"]] = cached
            except Exception:
                pass
        if _scene_cache:
            log.info("Scene cache loaded: %d minds from DB", len(_scene_cache))
    except Exception as e:
        log.debug("Scene cache load skipped: %s", e)


def _save_scene_to_db(mind_key: str, setting: str) -> None:
    """Persist scene setting into originals.meta_json for future reuse."""
    db_path = os.getenv("KD_DB_PATH", "") or os.path.join(
        os.path.dirname(__file__), "data", "keepsake.sqlite"
    )
    if not os.path.exists(db_path):
        return
    try:
        import sqlite3, json as _json
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT meta_json FROM originals WHERE temporality = ?", (mind_key,)
        ).fetchone()
        meta = _json.loads(row["meta_json"] or "{}") if row and row["meta_json"] else {}
        meta["scene_setting"] = setting
        meta["scene_setting_updated"] = _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())
        conn.execute(
            "UPDATE originals SET meta_json = ? WHERE temporality = ?",
            (_json.dumps(meta, ensure_ascii=False), mind_key),
        )
        conn.commit()
        conn.close()
        log.info("Scene setting cached to DB for %s", mind_key)
    except Exception as e:
        log.debug("Scene cache save failed for %s: %s", mind_key, e)


# Load cache at import time
_load_scene_cache_from_db()


def _extract_scene_setting(drift_text: str, mind_key: str) -> str:
    """Return the physical setting for a mind's memory.

    Each tick re-extracts from the current drift text so the scene evolves
    as the memory drifts. Falls back to simple keyword extraction if the
    LLM call fails.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return _fallback_setting(drift_text)

    excerpt = _sanitize_excerpt(_truncate_drift(drift_text, max_words=150))

    try:
        client = OpenAI(timeout=20.0)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=120,
            temperature=0.3,
            messages=[{
                "role": "user",
                "content": (
                    "Read this memory text and extract ONLY the physical setting.\n\n"
                    "Return 2-3 SHORT sentences describing the SPACE:\n"
                    "- FIRST: Is it INDOOR or OUTDOOR? State this clearly.\n"
                    "- What SPECIFIC type of place? (kitchen, garden, farm road, "
                    "allotment, shoreline, market, corridor, balcony, courtyard, "
                    "highway, field, rooftop, etc.) Be specific — not just 'a room'.\n"
                    "- What is the quality of light? (sunlight, overcast, fluorescent, "
                    "golden hour, night, rain-diffused, etc.)\n"
                    "- What are the materials, textures, surfaces visible?\n"
                    "- What is the weather/climate? (tropical, rainy, dry heat, etc.)\n\n"
                    "CRITICAL: If the memory describes an OUTDOOR place (garden, farm, "
                    "road, field, allotment, shoreline, market, street), the setting "
                    "MUST be outdoor. Do NOT convert outdoor settings into interiors.\n\n"
                    "STRIP OUT completely:\n"
                    "- All narrative events and actions\n"
                    "- All specific objects being used (ashes, letters, food, bags, etc.)\n"
                    "- All people, names, relationships, body parts\n"
                    "- All emotions, feelings, metaphors\n"
                    "- Do NOT mention death, burial, loss, or any story elements\n\n"
                    f"Memory:\n{excerpt}\n\n"
                    "Physical setting only (2-3 sentences):"
                ),
            }],
        )
        setting = resp.choices[0].message.content.strip() if resp.choices else ""
        if setting and len(setting) > 20:
            log.info("Extracted scene setting for %s: %s", mind_key, setting[:80])
            return setting
        return _fallback_setting(drift_text)
    except Exception as e:
        log.warning("Scene setting extraction failed for %s: %s", mind_key, e)
        return _fallback_setting(drift_text)


def preextract_scene_settings(memories: Dict[str, str]) -> Dict[str, str]:
    """Extract and cache scene settings for multiple memories at once.

    Called during bootstrap to pre-populate the scene cache so the first
    drift tick doesn't need to make 6 extra GPT-4o-mini calls.

    Args:
        memories: {mind_key: memory_text_en}

    Returns:
        {mind_key: scene_setting_text}
    """
    results: Dict[str, str] = {}
    for mind_key, text in memories.items():
        if not text or not text.strip():
            continue
        setting = _extract_scene_setting(text, mind_key)
        results[mind_key] = setting
    return results


def invalidate_scene_cache(mind_key: Optional[str] = None) -> None:
    """Clear cached scene settings (e.g. when original memories change).

    If mind_key is given, clears only that mind. Otherwise clears all.
    """
    global _scene_cache
    if mind_key:
        _scene_cache.pop(mind_key, None)
    else:
        _scene_cache.clear()
    log.info("Scene cache invalidated: %s", mind_key or "ALL")


def blend_scenes_batch(
    infiltrating_per_mind: Dict[str, List[str]],
    drift_texts: Optional[Dict[str, str]] = None,
    resonance_per_mind: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, str]]:
    """Blend base scene settings with infiltrating imagery for all minds.

    One batched GPT-4o-mini call that takes each mind's cached physical setting
    and the current tick's infiltrating_imagery, then outputs:
      - TRANSFORMED SCENE where the space has physically shifted
      - WEATHER derived from the semantic content of the pressure phrases
      - TIME OF DAY derived from the content's emotional/spatial quality
      - COLOR PALETTE derived from the content's material and tonal quality

    Every visual parameter is rooted in what was actually ingested — no
    artefacts from abstract score mappings. Resonance acts as an INTENSITY
    modifier: low resonance = muted/uncertain expression of the derived
    atmosphere; high resonance = vivid/confident expression.

    Args:
        infiltrating_per_mind: {mind_key: [phrase1, phrase2, phrase3]}
        drift_texts: {mind_key: drift_text} — used to extract base scene
            if not already cached.
        resonance_per_mind: {mind_key: float 0.0-1.0} — headline fit strength.
            Controls atmosphere intensity, not content.

    Returns:
        {mind_key: {"scene": str, "weather": str, "time_of_day": str,
                     "color_palette": str}}
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        log.warning("No OPENAI_API_KEY — skipping scene blending")
        return {}

    res_map = resonance_per_mind or {}

    # Collect base scenes (from cache) and imagery per mind
    entries = []
    mind_keys_ordered = []
    for mind_key, imagery in infiltrating_per_mind.items():
        if not imagery:
            continue
        # Get base scene from cache, or extract if needed
        base = _scene_cache.get(mind_key, "")
        if not base and drift_texts and drift_texts.get(mind_key):
            base = _extract_scene_setting(drift_texts[mind_key], mind_key)
        if not base:
            continue
        phrases = imagery[:3]
        res_val = res_map.get(mind_key, 0.5)
        # Format entry based on resonance level —
        # high resonance: pressure IS the scene (base hidden)
        # low resonance: base is the scene (pressure modifies it)
        pressure_block = "\n".join(f"  - {p}" for p in phrases)
        if res_val >= 0.6:
            # High resonance: pressure drives the scene, base is just faint context
            entries.append(
                f"MIND: {mind_key}\n"
                f"MODE: PRESSURE-DRIVEN (resonance {res_val:.2f} — the pressure "
                f"phrases define what the scene IS, not what it modifies)\n"
                f"SCENE SOURCE:\n{pressure_block}\n"
                f"FAINT ORIGIN (do NOT make this the subject — it's a ghost trace only): "
                f"{base[:80]}"
            )
        elif res_val >= 0.4:
            # Moderate: blend between base and pressure
            entries.append(
                f"MIND: {mind_key}\n"
                f"MODE: BLENDED (resonance {res_val:.2f} — the scene is a hybrid "
                f"between the base setting and the pressure environment)\n"
                f"BASE SETTING: {base}\n"
                f"PRESSURE:\n{pressure_block}"
            )
        else:
            # Low resonance: base dominates, pressure adds subtle shifts
            entries.append(
                f"MIND: {mind_key}\n"
                f"MODE: BASE-ANCHORED (resonance {res_val:.2f} — the base setting "
                f"remains dominant, pressure adds faint atmospheric shifts only)\n"
                f"BASE SETTING: {base}\n"
                f"PRESSURE:\n{pressure_block}"
            )
        mind_keys_ordered.append(mind_key)

    if not entries:
        return {}

    prompt = (
        "You generate scene descriptions and atmospheric conditions for photographs.\n\n"
        "CRITICAL RULE: NEVER translate pressure phrases literally into visual elements. "
        "'shattered calm' does NOT mean show broken things. 'slicing through routines' "
        "does NOT mean show cuts or fissures. 'crumbling safety' does NOT mean show "
        "rubble or ruins. Instead, find EVERYDAY SPACES whose spatial quality carries "
        "the same emotional register. 'shattered calm' → an empty hotel lobby at 3am. "
        "'absence slicing through routines' → a kitchen with the lights still on but "
        "no one there. The image must always show a NORMAL, INHABITABLE space — never "
        "disaster scenes, wastelands, ruins, or apocalyptic landscapes.\n\n"
        "For each mind below, produce 4 things. Every parameter must be TRACEABLE "
        "to the content — no arbitrary choices.\n\n"
        "Each mind has a MODE that controls how the scene is generated:\n\n"
        "**PRESSURE-DRIVEN** (high resonance): The SCENE SOURCE phrases define "
        "what kind of everyday space to depict — NOT by literal translation, but "
        "by finding a real, quiet, inhabitable place whose spatial quality matches "
        "the emotional register. Different from the origin but still a normal space. "
        "Examples:\n"
        "  - 'safety disrupted, familiar ground shifting' → an empty departure "
        "lounge, chairs still warm\n"
        "  - 'coastal erosion, salt wind, exposed roots' → a weathered waterfront "
        "promenade, salt-stained railings\n"
        "  - 'electric urgency, signals fracturing' → a server room corridor, "
        "blinking indicator lights\n"
        "  - 'weight of absence, routines dissolving' → a closed shopfront at "
        "dusk, display still lit\n"
        "The scene must be a place you could walk into. NEVER ruins, wastelands, "
        "cracked earth, apocalyptic voids, or disaster imagery.\n\n"
        "**BLENDED** (moderate resonance): The base setting is still present but "
        "the pressure shifts its character. A café under coastal pressure → the "
        "same café but with condensation on every surface, salt air quality, "
        "windows open to a changed sky. The space is recognisable but unsettled.\n\n"
        "**BASE-ANCHORED** (low resonance): The base setting dominates. "
        "Pressure adds faint atmospheric shifts only — lighting changes, "
        "surface textures warp slightly, but the space is still recognisably "
        "the same place.\n\n"
        "Output per mind:\n"
        "1. **scene**: 2-3 sentences describing a CONCRETE, INHABITABLE, EVERYDAY "
        "PHYSICAL SPACE. Follow the MODE strictly. For PRESSURE-DRIVEN mode, "
        "describe a different environment from the origin — but always a normal, "
        "real-world space (corridor, lobby, promenade, room, street, courtyard — "
        "NOT a wasteland, void, ruin, or abstract environment). NEVER mention the "
        "origin's type (café, balcony, car park, etc.) in the scene text.\n"
        "2. **weather**: 1 sentence. Derive weather/atmospheric conditions FROM the "
        "pressure content. Examples of content→weather reasoning:\n"
        "   - 'coastal restoration, mangrove roots' → humid salt haze, moisture\n"
        "   - 'data flows accelerating' → charged, static-electric air, dry clarity\n"
        "   - 'erosion, loss, wearing away' → wind-driven, dust suspended\n"
        "   - 'flooding, overflow' → rain, wet surfaces, saturated air\n"
        "   - 'heat, expansion, growth' → blazing dry heat, mirage shimmer\n"
        "   The weather must come FROM the content. INTENSITY modulates how "
        "strongly it manifests (low intensity = barely visible; high = dramatic).\n"
        "3. **time_of_day**: 1 short phrase. Derive from the content's quality:\n"
        "   - mechanical/technological content → artificial light, no natural time\n"
        "   - growth/renewal/beginning → early morning, dawn\n"
        "   - tension/peak/confrontation → midday, harsh overhead\n"
        "   - decay/ending/exhaustion → late afternoon, dusk\n"
        "   - stillness/absence/void → deep night\n"
        "   State the time AND describe its light quality in a few words.\n"
        "4. **color_palette**: 1 short phrase. Derive from the content's material quality:\n"
        "   - organic/natural → earth tones, greens, warm browns\n"
        "   - technological/digital → cool blues, whites, sharp contrast\n"
        "   - industrial/construction → steel grey, concrete, amber safety\n"
        "   - water/coastal → teal, grey-blue, salt-white\n"
        "   State the palette direction.\n\n"
        "Rules:\n"
        "- NO people, animals, or living beings in scenes (no silhouettes, no shadows of figures, no ghostly presences)\n"
        "- NO emotions or abstract metaphors — describe PHYSICAL conditions\n"
        "- Every parameter must be CAUSALLY CONNECTED to the pressure phrases\n"
        "- Low intensity = the derived conditions are faint/subtle; high = vivid/dramatic\n\n"
        + "\n\n".join(entries)
        + '\n\nReturn ONLY valid JSON:\n'
        '{"mind_key": {"scene": "...", "weather": "...", "time_of_day": "...", '
        '"color_palette": "..."}, ...}'
    )

    try:
        import json as _json
        client = OpenAI(timeout=30.0)
        t0 = _time.time()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=1200,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content.strip() if resp.choices else ""
        elapsed = _time.time() - t0
        log.info("Scene+atmosphere blending completed in %.1fs for %d minds",
                 elapsed, len(entries))

        # Log for LLM tracking
        if llm_logger:
            try:
                llm_logger.log_call(
                    provider="openai", model="gpt-4o-mini",
                    function="blend_scenes_batch",
                    prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                    completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
                    latency_ms=int(elapsed * 1000),
                )
            except Exception:
                pass

        # Parse JSON — handle markdown code fences
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        result = _json.loads(raw)

        # Validate: ensure each value is a dict with required keys
        blended: Dict[str, Dict[str, str]] = {}
        required_keys = {"scene", "weather", "time_of_day", "color_palette"}
        for mk in mind_keys_ordered:
            val = result.get(mk)
            if isinstance(val, dict):
                scene = val.get("scene", "")
                if isinstance(scene, str) and len(scene.strip()) > 20:
                    entry = {}
                    for k in required_keys:
                        v = val.get(k, "")
                        entry[k] = v.strip() if isinstance(v, str) and v.strip() else ""
                    if entry["scene"]:
                        blended[mk] = entry
                    else:
                        log.warning("Scene blend scene empty for %s", mk)
                else:
                    log.warning("Scene blend scene too short for %s", mk)
            elif isinstance(val, str) and len(val.strip()) > 20:
                # Backward compat: old format returned just a string
                blended[mk] = {"scene": val.strip(), "weather": "",
                                "time_of_day": "", "color_palette": ""}
                log.warning("Scene blend returned string for %s — no atmosphere", mk)
            else:
                log.warning("Scene blend missing/invalid for %s, using base", mk)
        return blended

    except Exception as e:
        log.warning("Scene+atmosphere blending failed: %s — falling back", e)
        return {}


def _fallback_setting(drift_text: str) -> str:
    """Simple fallback: extract place-like nouns from the text."""
    _place_words = _re.compile(
        r'\b(kitchen|garden|hallway|corridor|balcony|terrace|rooftop|'
        r'bedroom|bathroom|living\s+room|apartment|house|building|'
        r'street|road|alley|market|shop|store|mall|'
        r'shore|shoreline|beach|coast|sea|ocean|harbour|port|'
        r'desert|dune|sand|oasis|park|field|'
        r'mosque|church|school|hospital|office|warehouse|'
        r'window|doorway|staircase|elevator|lobby|courtyard|'
        r'bridge|overpass|underpass|tunnel|station|airport)\b',
        _re.IGNORECASE,
    )
    places = list(set(_place_words.findall(drift_text.lower())))[:3]
    if places:
        return f"A {', '.join(places)} space. Modern Gulf city setting, natural light."
    return "An interior space in a modern Gulf city. Natural ambient light, contemporary surfaces."


# ── Visual-condition derivation ──
# Translates abstract drift metadata (keywords, resonance, version)
# into CONCRETE photographic conditions (time, weather, camera, palette)
# so DALL-E produces visually distinct images of the same place each drift.

_TIME_OF_DAY = [
    "early dawn — first grey-blue light before sunrise, sky just brightening, "
    "cool quiet tones, surfaces catching the earliest light",
    "morning — soft directional sunlight streaming in, long gentle shadows, "
    "warm white light, fresh clear atmosphere",
    "midday — strong overhead sun, crisp shadows, bright surfaces, "
    "high-key lighting with sharp definition",
    "late afternoon — warm low-angle light, long golden shadows, "
    "rich warm tones, surfaces glowing where the light hits",
    "golden hour — deep amber light flooding the space, everything warmly lit, "
    "long dramatic shadows, saturated warm palette",
    "blue hour — sky turning deep indigo, artificial lights turning on, "
    "cool blue natural light mixing with warm artificial light",
    "dusk — sky pink and purple, last natural light fading, "
    "warm interior/artificial lights prominent, atmospheric transition",
    "overcast noon — bright but shadowless, even diffused white light, "
    "no directional shadows, soft and flat illumination, pastel sky",
]

_WEATHER_BY_RESONANCE = [
    # (threshold, description)
    (0.15, "dense fog — distances disappear, surfaces are wet, "
           "visibility under 20 metres, the space feels enclosed by mist"),
    (0.30, "light rain — wet surfaces reflecting light, droplets visible, "
           "sky grey and low, puddles forming on flat surfaces"),
    (0.45, "overcast haze — diffused flat light, no shadows, muted colours, "
           "sky white-grey, air feels thick and still"),
    (0.60, "thin cloud cover — soft light with occasional breaks, "
           "gentle shadows, slightly muted palette"),
    (0.75, "partly cloudy — patches of direct sun alternating with shadow, "
           "dynamic light, clouds visible in sky"),
    (0.90, "clear sky — crisp sharp light, deep shadows, vivid colours, "
           "full visibility, clean air"),
    (1.01, "brilliant clarity — crystal-clear atmosphere, intense light, "
           "every surface sharp and defined, saturated palette"),
]

_CAMERA_ANGLES = [
    "wide establishing shot — full scene visible, camera at medium distance, "
    "eye level, showing the space as a whole",
    "low angle — camera near ground level looking UP through the space, "
    "surfaces and structures loom above, sky prominent",
    "close-up detail — camera very close to ONE surface or texture "
    "that catches available light (concrete grain, glass reflection, "
    "tile edge, wet surface, leaf, sand ripple), "
    "shallow depth of field, rest of space softly blurred behind. "
    "The surface must be LIT and clearly visible",
    "high angle — camera looking DOWN into the scene from above, "
    "floor patterns and surfaces prominent, geometric composition",
    "oblique — camera at 45° angle across the space, depth through "
    "diagonal composition, leading lines pulling the eye through",
]

_COLOR_PALETTES = [
    "warm — amber, gold, terracotta, deep orange, soft yellows",
    "cool — steel blue, slate grey, pale cyan, cool white",
    "muted — desaturated earth tones, dusty rose, faded olive, grey-green",
    "high contrast — deep blacks against bright highlights, dramatic tonal range",
    "monochromatic warm — variations of a single warm hue (amber/sepia range)",
    "monochromatic cool — variations of a single cool hue (blue/teal range)",
    "natural — true-to-life colours as the scene would appear, ungraded",
    "split tone — warm highlights with cool shadows, cinematic colour grading",
]


def _derive_visual_conditions(
    drifted_keywords: Optional[List[str]],
    resonance: float,
    version: int,
    mind_key: str,
) -> Dict[str, str]:
    """Derive concrete photographic conditions from drift metadata.

    Deterministic (no API call) — uses keyword hashing, resonance mapping,
    and version cycling to produce concrete visual parameters that DALL-E
    can actually act on. Each drift naturally produces different conditions
    because the keywords and resonance change every tick.

    Returns {time_of_day, weather, camera, color_palette}.
    """
    kws = drifted_keywords or []
    kw_str = " ".join(sorted(str(k).lower() for k in kws))

    # Time of day: hash keywords + version for pseudo-random selection
    # Different keywords → different time of day
    kw_hash = hash(kw_str + str(version))
    time_of_day = _TIME_OF_DAY[kw_hash % len(_TIME_OF_DAY)]

    # Weather: directly mapped from resonance
    # Low resonance = uncertain/unstable = fog/rain
    # High resonance = confident/clear = crisp visibility
    weather = _WEATHER_BY_RESONANCE[-1][1]  # default: brilliant
    for threshold, desc in _WEATHER_BY_RESONANCE:
        if resonance < threshold:
            weather = desc
            break

    # Camera angle: cycles with version, offset by mind_key hash
    # so different minds get different angles on the same tick
    mind_offset = hash(mind_key) % len(_CAMERA_ANGLES)
    camera = _CAMERA_ANGLES[(version + mind_offset) % len(_CAMERA_ANGLES)]

    # Color palette: hash keywords differently from time
    # (shift by large prime so it decorrelates from time selection)
    palette_hash = hash(kw_str + str(version) + "palette_salt_7919")
    color_palette = _COLOR_PALETTES[palette_hash % len(_COLOR_PALETTES)]

    return {
        "time_of_day": time_of_day,
        "weather": weather,
        "camera": camera,
        "color_palette": color_palette,
    }


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
    prev_infiltrating_imagery: Optional[List[str]] = None,
    resonance: float = 0.5,
    headline_scene_subjects: Optional[List[str]] = None,
    drifted_keywords: Optional[List[str]] = None,
    invariables: Optional[List[dict]] = None,
    blended_scene: Optional[str] = None,
    visual_atmosphere: Optional[Dict[str, str]] = None,
    keepsake_en: Optional[str] = None,
) -> str:
    """
    Build an image prompt from drift state.

    Architecture:
    1. SCENE = the memory's physical place, TRANSFORMED by the emotional
       pressure of ingested content. If a blended_scene is provided (from
       blend_scenes_batch), it replaces the static cached setting — the
       space itself has shifted. If not, falls back to the cached setting.
    2. PHOTOGRAPHY DIRECTION = time of day, weather, camera, palette.
       When visual_atmosphere is provided (from blend_scenes_batch), weather,
       time_of_day, and color_palette are content-rooted — derived from the
       actual headline content by the LLM. Camera remains deterministic.
       Falls back to fully deterministic derivation when no LLM atmosphere.
    3. TEMPORAL LENS = quality of photographic attention (cycles per version).
    """
    # Rotate through lens qualities based on version
    perspectives = LENS_VISUAL.get(mind_key, LENS_VISUAL["human"])
    lens_visual = perspectives[version % len(perspectives)]

    # Load lens axes for visual deep-refraction
    try:
        import lens as _lens_mod
        ldef = _lens_mod.LENS_DEFINITIONS.get(mind_key, _lens_mod.LENS_DEFINITIONS.get("human", {}))
    except Exception:
        ldef = {}

    # Format invariables into subject anchors
    inv_lines = []
    if invariables:
        for inv in invariables[:6]:
            phrase = inv.get("phrase", "") if isinstance(inv, dict) else str(inv)
            cat = inv.get("category", "sensory") if isinstance(inv, dict) else "sensory"
            if cat != "proper_noun" and phrase:
                inv_lines.append(f"  - {phrase}")

    # ── Derive visual conditions ──
    # Camera angle is always deterministic (compositional, not content-linked)
    det_cond = _derive_visual_conditions(
        drifted_keywords=drifted_keywords,
        resonance=resonance,
        version=version,
        mind_key=mind_key,
    )

    # Build final visual_cond: prefer LLM-derived atmosphere when available
    visual_cond = dict(det_cond)  # start with deterministic defaults
    if visual_atmosphere:
        # Override with content-rooted parameters from blend_scenes_batch
        for key in ("weather", "time_of_day", "color_palette"):
            llm_val = visual_atmosphere.get(key, "")
            if llm_val:
                visual_cond[key] = llm_val
        log.debug("Using LLM-derived atmosphere for %s: weather=%s, time=%s, palette=%s",
                  mind_key, visual_cond["weather"][:40],
                  visual_cond["time_of_day"][:40], visual_cond["color_palette"][:40])

    parts = []

    # ── 0. ABSOLUTE RULES ──
    parts.extend([
        "ABSOLUTE RULES (override everything else):",
        "1. NO LIVING BEINGS — no people, silhouettes, figures, shadows of humans, "
        "ghosts, animals, birds, insects, creatures of any kind. The scene must be "
        "empty of any living or once-living form.",
        "2. NO LITERAL DEPICTION of narrative events. Do NOT show: ashes, burial, "
        "cooking, eating, letters, photographs, phones, screens, bags, luggage, "
        "or any specific object from a story. The image is ATMOSPHERIC, not illustrative.",
        "3. NO recognisable real places, landmarks, or named buildings.",
        "4. The image MUST be a COMPLETE photographic scene — no blank/white areas, "
        "no transparency, no abstract compositions. A real, full photograph.",
        "5. Do NOT interpret ANY word literally. Not emotional words, not narrative "
        "words, not object words from the memory. The image captures a FEELING "
        "through the quality of space, light, and atmosphere — never through "
        "depicting specific things mentioned in the text.",
        "",
    ])

    # ── 1. SCENE — the physical place, transformed by ingested pressure ──
    if blended_scene:
        # Blended scene: the space has already been transformed by GPT-4o-mini
        # based on the base setting + infiltrating_imagery
        parts.extend([
            "SCENE (the physical space — transformed by external pressure):",
            f'"{blended_scene}"',
            "",
            f"GEOLOCATION: {_INSTANCE_GEO}",
            "",
            "Photograph this transformed space exactly as described. "
            "The space has shifted from its original form — render the "
            "transformation faithfully. If it describes dissolved boundaries, "
            "extended corridors, fogged glass, shifted surfaces — show that.",
            "",
            "If the scene is OUTDOOR, photograph it as a full OUTDOOR scene — "
            "sky, vegetation, ground, weather. Do NOT place it inside a building.",
            "",
        ])
    else:
        # Fallback: static cached setting (no blending available)
        scene_setting = _extract_scene_setting(drift_text, mind_key)
        parts.extend([
            "LOCATION (the physical place — what the camera points at):",
            f'"{scene_setting}"',
            "",
            f"GEOLOCATION: {_INSTANCE_GEO}",
            "",
            "If the setting is OUTDOOR (garden, farm, field, road, allotment, "
            "shoreline, market, street), photograph it as a full OUTDOOR scene — "
            "sky, vegetation, ground, weather. Do NOT place it inside a building.",
            "",
        ])

    # ── 2. PHOTOGRAPHY DIRECTION — concrete conditions that change per drift ──
    parts.extend([
        "PHOTOGRAPHY DIRECTION (these are MANDATORY — follow each exactly):",
        "",
        f"TIME OF DAY: {visual_cond['time_of_day']}",
        "Render the lighting EXACTLY as this time of day would appear. "
        "The direction, colour, and intensity of light must match this time.",
        "",
        f"WEATHER: {visual_cond['weather']}",
        "The atmospheric conditions MUST be visible in the image — they affect "
        "sky, surfaces, air clarity, reflections, and material appearance.",
        "",
        f"CAMERA: {visual_cond['camera']}",
        "Frame and compose the photograph exactly as described. "
        "The camera position and framing are as important as the subject.",
        "",
        f"COLOR PALETTE: {visual_cond['color_palette']}",
        "The overall colour grading of the photograph should lean toward this palette.",
        "",
    ])

    # ── 3. SENSORY ANCHORS — invariable textures from the original memory ──
    if inv_lines:
        parts.extend([
            "SENSORY DETAILS in the scene (textures, materials — NOT objects or events):",
            *inv_lines,
            "",
        ])

    # ── 4. TEMPORAL LENS + LENS AXES ──
    if ldef:
        parts.extend([
            "LENS AXES (govern what the camera attends to, compositional weight, and implied forces):",
            f"  Object of attention: {ldef.get('object_of_attention', '')}",
            f"    → The camera is drawn to subjects from this world. Let the image be OF this domain.",
            f"  Temporal scale: {ldef.get('temporal_scale', '')}",
            f"    → The composition should carry the weight of this duration.",
            f"  Causal structure: {ldef.get('causal_structure', '')}",
            f"    → The relationship between elements should imply this kind of force at work.",
            f"  Scene framing: {lens_visual}",
            "Apply as photographic quality — depth of field, focus behaviour, light character.",
            "",
        ])
    else:
        parts.extend([
            f"TEMPORAL LENS: {lens_visual}",
            "Apply as photographic quality — depth of field, focus behaviour, light character.",
            "",
        ])

    # ── Museum narration alignment ──
    if keepsake_en and keepsake_en.strip():
        parts.extend([
            "MUSEUM NARRATION (text displayed alongside this image — the image should belong to the same visual world):",
            keepsake_en.strip()[:400],
            "",
        ])

    # ── 5. VISUAL STYLE ──
    parts.extend([
        "VISUAL STYLE: Contemporary photorealistic photography. Clean, modern. "
        "NOT vintage, NOT film grain, NOT sepia, NOT grunge, NOT decayed. "
        "No cracked walls, no peeling paint, no rust, no ruins. "
        "Photograph this place as if you arrived when no one was there. "
        "No event is taking place — only the space, its light, its weather.",
        "",
    ])

    # ── 6. GUARDRAILS ──
    parts.extend([
        "Portrait (2:3). No text, logos, or writing. No transparency.",
        "NO LIVING BEINGS of any kind.",
        "NO narrative objects (ashes, letters, photographs, food, bags, phones, screens).",
        "No religious, political, or military imagery.",
        "No recognisable real places or landmarks.",
    ])

    return "\n".join(parts)


def _image_path(mind_key: str, version: int) -> Path:
    """Return the local file path for a cached image."""
    return IMAGE_DIR / f"{mind_key}_v{version}.png"


def _quality_stamp_path(img_path: Path) -> Path:
    """Sidecar file that records which model+quality generated an image."""
    return img_path.with_suffix(".gen")


def _quality_matches_current(img_path: Path) -> bool:
    """Return True only if the cached image was generated with the current model+quality."""
    try:
        stamp = _quality_stamp_path(img_path).read_text().strip()
        return stamp == f"{IMAGE_MODEL}|{IMAGE_QUALITY}"
    except Exception:
        return False  # no stamp = generated before this check existed → treat as stale


def _write_quality_stamp(img_path: Path) -> None:
    """Write model+quality stamp next to the saved image."""
    try:
        _quality_stamp_path(img_path).write_text(f"{IMAGE_MODEL}|{IMAGE_QUALITY}")
    except Exception as e:
        log.warning("Failed to write quality stamp for %s: %s", img_path.name, e)


def _fallback_to_previous_image(mind_key: str, version: int) -> Path | None:
    """
    Never copy a previous image — return None so the page falls back to
    its own generic display rather than silently reusing stale art.
    A .fallback marker is left so the next tick knows to regenerate.
    """
    cur = _image_path(mind_key, version)
    cur.with_suffix(".fallback").touch()
    log.warning("Image generation exhausted for %s v%d — NOT copying previous image. Marked for regeneration.", mind_key, version)
    return None


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
    prev_infiltrating_imagery: Optional[List[str]] = None,
    resonance: float = 0.5,
    headline_scene_subjects: Optional[List[str]] = None,
    drifted_keywords: Optional[List[str]] = None,
    invariables: Optional[List[dict]] = None,
    blended_scene: Optional[str] = None,
    visual_atmosphere: Optional[Dict[str, str]] = None,
    keepsake_en: Optional[str] = None,
) -> Optional[str]:
    """
    Generate an image for a temporality's current drift state.

    Default: gpt-image-1-mini (low, 1024x1536) — ~$0.005-0.01/image.

    blended_scene: if provided, replaces the static cached scene setting
    with a dynamically transformed version where the space has been shifted
    by the emotional pressure of the current tick's ingested content.

    visual_atmosphere: if provided, contains content-rooted weather,
    time_of_day, and color_palette derived from the actual headline content
    by the LLM (via blend_scenes_batch). Overrides deterministic derivation
    for those parameters. Camera angle remains deterministic.

    Returns a tuple (local_file_path, prompt_text) if successful,
    or (None, "") on failure. When cached, returns (path, "") since
    the prompt is not stored alongside the cached file.
    """
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    out_path = _image_path(mind_key, version)

    # Already cached — skip only if quality settings haven't changed
    fallback_marker = out_path.with_suffix(".fallback")
    if out_path.exists() and out_path.stat().st_size > 0 and not fallback_marker.exists():
        if _quality_matches_current(out_path):
            log.info("Image already cached at current quality (%s/%s): %s",
                     IMAGE_MODEL, IMAGE_QUALITY, out_path)
            return str(out_path), ""
        else:
            log.info("Quality settings changed → regenerating %s (was: %s, now: %s/%s)",
                     out_path.name,
                     _quality_stamp_path(out_path).read_text().strip()
                     if _quality_stamp_path(out_path).exists() else "unknown",
                     IMAGE_MODEL, IMAGE_QUALITY)
            out_path.unlink(missing_ok=True)
            _quality_stamp_path(out_path).unlink(missing_ok=True)
    # Remove fallback marker if present — we're regenerating now
    fallback_marker.unlink(missing_ok=True)

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
        prev_infiltrating_imagery=prev_infiltrating_imagery,
        resonance=resonance,
        drifted_keywords=drifted_keywords,
        invariables=invariables,
        headline_scene_subjects=headline_scene_subjects,
        blended_scene=blended_scene,
        visual_atmosphere=visual_atmosphere,
        keepsake_en=keepsake_en,
    )

    log.info("Generating image for %s v%d (model=%s, size=%s)", mind_key, version, IMAGE_MODEL, IMAGE_SIZE)

    for attempt in range(1, MAX_REGEN_ATTEMPTS + 1):
        t0 = _time.time()
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
                _write_quality_stamp(out_path)

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
                _write_quality_stamp(out_path)

            file_size_kb = out_path.stat().st_size / 1024
            latency = int((_time.time() - t0) * 1000)
            log.info("Saved image: %s (%.1f KB, attempt %d)", out_path, file_size_kb, attempt)

            # ── Extract token usage from response ──
            _tokens_in = 0
            _tokens_out = 0
            try:
                if hasattr(resp, 'usage') and resp.usage:
                    _tokens_in = getattr(resp.usage, 'input_tokens', 0) or 0
                    _tokens_out = getattr(resp.usage, 'output_tokens', 0) or 0
                    # Some models use 'prompt_tokens' / 'completion_tokens' instead
                    if _tokens_in == 0:
                        _tokens_in = getattr(resp.usage, 'prompt_tokens', 0) or 0
                    if _tokens_out == 0:
                        _tokens_out = getattr(resp.usage, 'completion_tokens', 0) or 0
            except Exception:
                pass

            if llm_logger:
                llm_logger.log_call(
                    operation="image_gen", backend="openai", model=IMAGE_MODEL,
                    latency_ms=latency, success=True,
                    extra=f"{mind_key}_v{version}",
                    tokens_in=_tokens_in, tokens_out=_tokens_out,
                )

            # ── Quality gate: reject blown-out / mostly-black images ──
            if not _image_quality_ok(out_path):
                qg_reason = _image_quality_reason(out_path)
                log.warning(
                    "Image quality gate FAILED for %s v%d (attempt %d/%d) — %s",
                    mind_key, version, attempt, MAX_REGEN_ATTEMPTS,
                    "regenerating..." if attempt < MAX_REGEN_ATTEMPTS else "falling back to previous version",
                )
                # Log the quality gate rejection so it shows in admin
                if llm_logger:
                    llm_logger.log_call(
                        operation="image_regen", backend="openai", model=IMAGE_MODEL,
                        latency_ms=0, success=False,
                        error_msg=f"Quality gate: {qg_reason}",
                        extra=f"{mind_key}_v{version} attempt={attempt}/{MAX_REGEN_ATTEMPTS}",
                    )
                if attempt < MAX_REGEN_ATTEMPTS:
                    out_path.unlink(missing_ok=True)
                    continue  # retry
                # Last attempt — fall back to previous version's image
                out_path.unlink(missing_ok=True)
                prev_img = _fallback_to_previous_image(mind_key, version)
                if prev_img:
                    log.warning("Quality gate: using fallback image %s for %s v%d", prev_img, mind_key, version)
                    return str(prev_img), prompt
                log.error("Quality gate: no fallback image available for %s v%d", mind_key, version)
                return None, prompt

            # ── Content gate: reject images with living beings or bad quality ──
            content_ok, content_reason = _image_content_ok(out_path)
            if not content_ok:
                # Distinguish living-beings (hard reject) from quality issues (soft)
                _has_living = any(w in content_reason.lower() for w in [
                    "person", "people", "human", "figure", "silhouette", "face",
                    "hand", "animal", "bird", "flamingo", "fish", "insect", "creature",
                    "ghost",
                ])
                log.warning(
                    "Content gate FAILED for %s v%d (attempt %d/%d): %s — %s",
                    mind_key, version, attempt, MAX_REGEN_ATTEMPTS, content_reason,
                    "regenerating..." if attempt < MAX_REGEN_ATTEMPTS else (
                        "hard reject (living beings)" if _has_living else "accepting best effort (quality)"
                    ),
                )
                if llm_logger:
                    llm_logger.log_call(
                        operation="image_regen", backend="openai", model=IMAGE_MODEL,
                        latency_ms=0, success=False,
                        error_msg=f"Content gate: {content_reason}",
                        extra=f"{mind_key}_v{version} attempt={attempt}/{MAX_REGEN_ATTEMPTS}",
                    )
                if attempt < MAX_REGEN_ATTEMPTS:
                    out_path.unlink(missing_ok=True)
                    continue  # retry
                # Last attempt: reject and fall back to previous version
                out_path.unlink(missing_ok=True)
                if _has_living:
                    log.warning("Content gate: REJECTING %s v%d — living beings after %d attempts", mind_key, version, attempt)
                else:
                    log.warning("Content gate: REJECTING %s v%d — quality issue after %d attempts", mind_key, version, attempt)
                prev_img = _fallback_to_previous_image(mind_key, version)
                if prev_img:
                    log.warning("Content gate: using fallback image %s for %s v%d", prev_img, mind_key, version)
                    return str(prev_img), prompt
                return None, prompt

            # Log revised prompt if available (DALL-E 3 revises prompts)
            revised = getattr(resp.data[0], "revised_prompt", None) if resp.data else None
            if revised:
                log.debug("Revised prompt for %s v%d: %s", mind_key, version, revised[:200])

            return str(out_path), prompt

        except Exception as e:
            latency = int((_time.time() - t0) * 1000)
            log.error("Image generation failed for %s v%d (attempt %d): %s", mind_key, version, attempt, e)

            if llm_logger:
                llm_logger.log_call(
                    operation="image_gen", backend="openai", model=IMAGE_MODEL,
                    latency_ms=latency, success=False,
                    error_msg=f"{type(e).__name__}: {str(e)[:200]}",
                    extra=f"{mind_key}_v{version}",
                    tokens_in=0, tokens_out=0,
                )

            # Clean up partial file
            if out_path.exists():
                try:
                    out_path.unlink()
                except Exception:
                    pass
            return None, prompt

    return None, prompt  # shouldn't reach here, but just in case


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
