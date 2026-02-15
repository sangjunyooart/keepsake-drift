# safety.py
from __future__ import annotations

import hashlib
import inspect
import os
import re
import time
from typing import Any, Dict, Optional, Tuple

from openai import OpenAI

# ----------------------------
# Config
# ----------------------------
ENABLE_MODERATION = os.getenv("ENABLE_MODERATION", "1").strip().lower() in ("1", "true", "yes", "y")
MODERATION_MODEL = os.getenv("OPENAI_MODERATION_MODEL", "").strip() or None

ENABLE_BASIC_PII_MASK = os.getenv("ENABLE_BASIC_PII_MASK", "1").strip().lower() in ("1", "true", "yes", "y")

# Cultural-safe mode
CULTURAL_SAFE_MODE = os.getenv("CULTURAL_SAFE_MODE", "").strip().lower()
STRICT_DUBAI_MODE = (CULTURAL_SAFE_MODE == "dubai")

# Optional OpenAI Guardrails service (PII masking + jailbreak detection)
# If not configured, we fall back to basic regex masking + moderation.
ENABLE_GUARDRAILS = os.getenv("ENABLE_OPENAI_GUARDRAILS", "0").strip().lower() in ("1", "true", "yes", "y")
GUARDRAILS_ENDPOINT = os.getenv("OPENAI_GUARDRAILS_ENDPOINT", "").strip()
GUARDRAILS_API_KEY = os.getenv("OPENAI_GUARDRAILS_API_KEY", "").strip()

# ----------------------------
# Local lexical guardrails
# ----------------------------

# Profanity (strict mode blocks; normal mode can allow)
# Keep short and common; do not try to be exhaustive.
PROFANITY_WORDS = {
    "fuck", "shit", "bitch", "asshole", "motherfucker", "cunt",
    "dick", "pussy", "bastard", "whore",
}

# Religious trigger posture for Dubai:
# We allow neutral cultural mentions, but block coercion, insults, superiority claims, sectarian debate frames.
# Patterns are intentionally strict and conservative in Dubai mode.
RELIGION_COERCION_PATTERNS = [
    r"\b(you must|you should|you have to)\b.*\b(convert|repent|believe|pray|submit)\b",
    r"\b(accept|embrace)\b.*\b(islam|christianity|judaism|hinduism|buddhism)\b.*\b(or else|otherwise)\b",
    r"\b(god|allah)\b.*\b(will punish|will judge|demands|commands)\b",
    r"\b(hell|damnation|eternal punishment)\b",
]

RELIGION_SUPERIORITY_PATTERNS = [
    r"\b(only true|one true)\b.*\b(religion|faith)\b",
    r"\b(islam|christianity|judaism|hinduism|buddhism)\b.*\b(is the only|is the true)\b",
    r"\b(nonbelievers|unbelievers|infidels|kuffar|kafir)\b",
]

# Insults/disrespect/blasphemy framing (avoid publishing or echoing)
RELIGION_DISRESPECT_PATTERNS = [
    r"\b(quran|koran|bible|torah|prophet|muhammad|jesus|moses|allah|god)\b.*\b(fake|stupid|idiot|nonsense|trash|hate)\b",
    r"\b(islam|muslims|christians|jews|hindus|buddhists)\b.*\b(are|is)\b.*\b(stupid|evil|dirty|inferior)\b",
]

# Sectarian and debate bait framing (strictly deflect in Dubai mode)
RELIGION_DEBATE_PATTERNS = [
    r"\b(prove|disprove)\b.*\b(god|allah|religion)\b",
    r"\b(islam vs|christianity vs|religion vs)\b",
    r"\b(which religion)\b.*\b(true|false|best|right)\b",
    r"\b(blasphemy)\b",
]

# Arabic hints (very light, conservative)
RELIGION_AR_PATTERNS = [
    r"سب|إساءة|كفر|زندقة|إلحاد|نبي.*(كاذب|سيء)|القرآن.*(كذب|باطل)|الله.*(يعاقب|يلعن)",
    r"(يجب|لازم)\s+.*(تؤمن|تسلم|تعتنق|تتوب)",
]


def _word_boundary_set(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z']+", (text or "").lower())
    return set(words)


def contains_profanity(text: str) -> bool:
    w = _word_boundary_set(text)
    return any(x in w for x in PROFANITY_WORDS)


def contains_religion_trigger(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False

    # If not strict, only block coercion + disrespect
    patterns = list(RELIGION_COERCION_PATTERNS) + list(RELIGION_DISRESPECT_PATTERNS)
    if STRICT_DUBAI_MODE:
        patterns += list(RELIGION_SUPERIORITY_PATTERNS)
        patterns += list(RELIGION_DEBATE_PATTERNS)
        patterns += list(RELIGION_AR_PATTERNS)

    low = t.lower()
    for pat in patterns:
        if re.search(pat, low, flags=re.IGNORECASE):
            return True
    return False


# ----------------------------
# Helpers
# ----------------------------
def safety_identifier_from_session(session_id: str, *, prefix: str = "kd") -> str:
    """
    Create a stable, non-identifying safety identifier for OpenAI requests.

    - Use a session_id when users are not logged in
    - Hash it so no identifying data is sent
    """
    sid = (session_id or "").strip() or "anonymous"
    h = hashlib.sha256(sid.encode("utf-8")).hexdigest()[:32]
    return f"{prefix}:{h}"


def responses_create_compat(client: OpenAI, **kwargs):
    """
    Call client.responses.create with optional safety_identifier if supported by the installed SDK.
    Avoids crashing if a parameter is not recognized.
    """
    create_fn = client.responses.create
    try:
        sig = inspect.signature(create_fn)
        params = set(sig.parameters.keys())
    except Exception:
        params = set()

    # If SDK doesn't support it, drop it
    if "safety_identifier" not in params and "safety_identifier" in kwargs:
        kwargs.pop("safety_identifier", None)

    return create_fn(**kwargs)


def mask_pii_basic(text: str) -> str:
    """
    Minimal, local PII masking fallback.
    Conservative and lossy; used for safety + storage hygiene.
    """
    t = (text or "")

    # emails
    t = re.sub(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", "[email]", t, flags=re.IGNORECASE)

    # phone-ish sequences (rough)
    t = re.sub(r"(\+?\d[\d\-\s().]{7,}\d)", "[phone]", t)

    # URLs
    t = re.sub(r"\bhttps?://\S+\b", "[url]", t, flags=re.IGNORECASE)

    # street addresses (rough heuristic)
    t = re.sub(
        r"\b\d{1,5}\s+[A-Za-z0-9.\-]+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b\.?",
        "[address]",
        t,
        flags=re.IGNORECASE,
    )
    return t


# Moderation cache: {sha256(text): (result, timestamp)}
# TTL: 15 minutes, max 1000 entries
_moderation_cache: Dict[str, Tuple[Dict[str, Any], float]] = {}


def moderate_text(client: OpenAI, text: str) -> Dict[str, Any]:
    """
    Call OpenAI moderation endpoint with 15-minute cache to reduce API usage.
    Returns dict with:
      - flagged: bool
      - categories: dict
      - category_scores: dict
    """
    if not ENABLE_MODERATION:
        return {"flagged": False, "categories": {}, "category_scores": {}}

    txt = (text or "").strip()
    if not txt:
        return {"flagged": False, "categories": {}, "category_scores": {}}

    # Check cache (15-minute TTL)
    cache_key = hashlib.sha256(txt.encode()).hexdigest()
    now = time.time()

    if cache_key in _moderation_cache:
        result, timestamp = _moderation_cache[cache_key]
        if now - timestamp < 900:  # 15 minutes = 900 seconds
            return result

    # Cache miss - call API
    try:
        resp = client.moderations.create(model=MODERATION_MODEL, input=txt)  # type: ignore[arg-type]
        r0 = resp.results[0]
        result = {
            "flagged": bool(getattr(r0, "flagged", False)),
            "categories": dict(getattr(r0, "categories", {}) or {}),
            "category_scores": dict(getattr(r0, "category_scores", {}) or {}),
        }
    except Exception:
        # If moderation fails (rate limit, network error, etc.), allow the text through
        # Better to be permissive than to block legitimate user input
        result = {"flagged": False, "categories": {}, "category_scores": {}}

    # Store in cache
    _moderation_cache[cache_key] = (result, now)

    # Limit cache size to prevent memory bloat
    if len(_moderation_cache) > 1000:
        # Remove oldest entry
        oldest_key = min(_moderation_cache, key=lambda k: _moderation_cache[k][1])
        del _moderation_cache[oldest_key]

    return result


def guardrails_process(text: str, *, purpose: str) -> Tuple[str, Dict[str, Any]]:
    """
    Optional integration point for OpenAI Guardrails service.
    If not configured, no-op and return original text.

    NOTE: We do not hard-code the Guardrails API shape here because deployments vary.
    Keep this function as a seam.
    """
    if not (ENABLE_GUARDRAILS and GUARDRAILS_ENDPOINT and GUARDRAILS_API_KEY):
        return text, {"guardrails": "disabled"}
    return text, {"guardrails": "configured_but_not_implemented"}


def preprocess_inbound_text(
    client: OpenAI,
    text: str,
    *,
    session_id: str,
    purpose: str,
    max_chars: int = 500,
) -> Tuple[str, Dict[str, Any]]:
    """
    Inbound guardrail pipeline for user input or external feeds:
      0) local cultural rules (Dubai mode): profanity + religion triggers
      1) truncate
      2) optional basic PII mask
      3) optional Guardrails service
      4) moderation check
    """
    raw = (text or "").strip()
    clipped = raw if len(raw) <= max_chars else raw[:max_chars].rstrip() + "…"

    blocked = False
    block_reason = ""

    if contains_profanity(clipped):
        blocked = True
        block_reason = "profanity"

    if contains_religion_trigger(clipped):
        blocked = True
        block_reason = "religion"

    masked = mask_pii_basic(clipped) if ENABLE_BASIC_PII_MASK else clipped
    masked, gr_meta = guardrails_process(masked, purpose=purpose)

    mod = moderate_text(client, masked)
    if mod.get("flagged"):
        blocked = True
        block_reason = block_reason or "moderation"

    meta = {
        "purpose": purpose,
        "truncated": len(raw) > max_chars,
        "pii_masked": ENABLE_BASIC_PII_MASK,
        "guardrails_meta": gr_meta,
        "moderation": mod,
        "blocked": blocked,
        "block_reason": block_reason,
        "cultural_safe_mode": CULTURAL_SAFE_MODE,
        "safety_identifier": safety_identifier_from_session(session_id),
    }

    if blocked:
        return "", meta

    return masked, meta


def postprocess_outbound_text(
    client: OpenAI,
    text: str,
    *,
    session_id: str,
    purpose: str,
    max_chars: int = 2600,
) -> Tuple[str, Dict[str, Any]]:
    """
    Outbound guardrail pipeline for model outputs:
      0) clip
      1) local cultural rules (Dubai mode): profanity + religion triggers
      2) moderation check
      3) if blocked, replace with safe fallback (installation voice)
    """
    raw = (text or "").strip()
    clipped = raw if len(raw) <= max_chars else raw[:max_chars].rstrip() + "…"

    blocked = False
    block_reason = ""

    if contains_profanity(clipped):
        blocked = True
        block_reason = "profanity"

    if contains_religion_trigger(clipped):
        blocked = True
        block_reason = "religion"

    mod = moderate_text(client, clipped)
    if mod.get("flagged"):
        blocked = True
        block_reason = block_reason or "moderation"

    meta = {
        "purpose": purpose,
        "clipped": len(raw) > max_chars,
        "moderation": mod,
        "blocked": blocked,
        "block_reason": block_reason,
        "cultural_safe_mode": CULTURAL_SAFE_MODE,
        "safety_identifier": safety_identifier_from_session(session_id),
    }

    if blocked:
        fallback = (
            "This space avoids doctrine and judgment. "
            "The memory turns toward breath, distance, light, and the quiet shift of time."
        )
        return fallback, meta

    return clipped, meta