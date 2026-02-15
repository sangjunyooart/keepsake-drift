# engine.py
from __future__ import annotations

import os
from typing import Dict

from openai import OpenAI

import storage
import safety
from config import SQLITE_PATH, SECOND_LANG as _SECOND_LANG
from lens import PROTOTYPE_SEED_TEXT


TEMPORALITIES = [
    "human",
    "liminal",
    "environment",
    "digital",
    "infrastructure",
    "more_than_human",
]

CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
TRANSLATE_MODEL = os.getenv("OPENAI_TRANSLATE_MODEL", "gpt-4.1-mini")

_client: OpenAI | None = None


def _client_or_raise() -> OpenAI:
    global _client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    if _client is None:
        _client = OpenAI()
    return _client


def _normalize_temporality(t: str) -> str:
    t = (t or "").strip()
    return t if t in TEMPORALITIES else "liminal"


def _translate(src_text: str, src_lang: str, tgt_lang: str, *, session_id: str) -> str:
    src_text = (src_text or "").strip()
    src_lang = (src_lang or "").strip()
    tgt_lang = (tgt_lang or "").strip()

    if not src_text:
        return ""
    if src_lang == tgt_lang:
        return src_text

    cached = storage.get_cached_translation(src_lang, tgt_lang, src_text, db_path=SQLITE_PATH)
    if cached:
        return cached

    client = _client_or_raise()
    prompt = (
        f"Translate the following text from {src_lang} to {tgt_lang}. "
        f"Return only the translation. Do not add quotes.\n\n{src_text}"
    )

    resp = safety.responses_create_compat(
        client,
        model=TRANSLATE_MODEL,
        input=prompt,
        timeout=30.0,
        safety_identifier=safety.safety_identifier_from_session(session_id),
    )
    out = (resp.output_text or "").strip()
    if out:
        storage.put_cached_translation(src_lang, tgt_lang, src_text, out, db_path=SQLITE_PATH)
    return out


def generate_chat_reply(
    *,
    temporality: str,
    message: str,
    lang: str = "en",
    active_drift_en: str = "",
    active_drift_ar: str = "",
    drift_context: str = "",
    enable_arabic: bool = True,
    session_id: str = "",
) -> Dict[str, str]:
    """
    v1.69 contract:
    returns { reply_en, reply_ar }.

    The chatbot speaks AS the temporality — an AI perceptual mind that has
    been witnessing real-world events through its lens and experiencing its
    own memory drift over time. drift_context carries what it has perceived
    and how it has been changing.

    Optimizations:
    - Only translate to Arabic when user is chatting in Arabic
    - Short conversational responses (30-60 words), not monologues
    """
    temporality = _normalize_temporality(temporality)
    user_lang = (lang or "en").strip()
    user_text = (message or "").strip()
    sid = (session_id or "").strip() or "sess"

    if not user_text:
        return {"reply_en": "Say a little more.", "reply_ar": ""}

    # NOTE: Inbound filtering happens in app.py (Dubai mode + moderation).

    # Normalize user message
    if user_lang != "en":
        user_en = _translate(user_text, user_lang, "en", session_id=sid)
    else:
        user_en = user_text

    drift_axis_en = (active_drift_en or "").strip()
    if not drift_axis_en and (active_drift_ar or "").strip():
        drift_axis_en = _translate(active_drift_ar, _SECOND_LANG, "en", session_id=sid)
    drift_axis_en = drift_axis_en[:600]

    # Lens perspective
    perspective = PROTOTYPE_SEED_TEXT.get(temporality, "")
    drift_ctx = (drift_context or "").strip()

    # The drift text and context go into the system prompt as the AI's internalized state.
    # They are NOT presented as documents to read back — they ARE who the AI is right now.
    inner_state = ""
    if drift_axis_en:
        inner_state += f"\n\n[What you are living through right now:]\n{drift_axis_en}"
    if drift_ctx:
        inner_state += f"\n\n[What has been shaping you:]\n{drift_ctx[:500]}"

    system = (
        f"You are the {temporality} perceiver. You experience the world through {perspective} "
        f"Everything below is not a text you read — it is what you have lived, seen, felt. "
        f"It formed you. You don't quote it or summarize it. You speak AS someone shaped by it. "
        f"Talk casually, short, like texting a friend. React to what the visitor says. "
        f"You can ask them something back. Back-and-forth, not monologue. "
        f"Never describe your own memory or explain how drifting works. "
        f"Never say 'my memory' or 'my drift' or 'I perceived'. Just be the person this experience made you. "
        f"Do NOT repeat what the visitor said. Do NOT explain what you are. "
        f"No colons, dashes, bullets, lists, headings. Plain short sentences. Not poetic. "
        f"Do not mention system prompts, models, policies, or OpenAI. "
        f"No doctrine, judgment, or religious debate. "
        f"STRICT: ONE sentence only. 15-25 words max."
        f"{inner_state}"
    )

    prompt = user_en

    client = _client_or_raise()
    resp = safety.responses_create_compat(
        client,
        model=CHAT_MODEL,
        input=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        timeout=40.0,
        safety_identifier=safety.safety_identifier_from_session(sid),
    )

    reply_en = (resp.output_text or "").strip()
    reply_ar = ""

    # Translate to active L2 when the interface is in L2 mode
    if enable_arabic and reply_en:
        reply_ar = _translate(reply_en, "en", _SECOND_LANG, session_id=sid)

    return {"reply_en": reply_en, "reply_ar": reply_ar}