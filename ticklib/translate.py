# ticklib/translate.py
from __future__ import annotations

import os
import sqlite3
from typing import Optional

from openai import OpenAI

import storage


def _openai_translate_en_to_ar(text_en: str, *, timeout_seconds: float = 40.0) -> str:
    """
    Minimal translation call (no response_format).
    """
    model = os.getenv("OPENAI_TRANSLATE_MODEL", os.getenv("OPENAI_DRIFT_MODEL", "gpt-4.1-mini"))
    client = OpenAI()

    prompt = (
        "Translate the following English into Arabic.\n"
        "Rules:\n"
        "- Output Arabic only.\n"
        "- Keep punctuation natural.\n"
        "- Keep proper nouns as commonly written in Arabic when appropriate.\n\n"
        f"English:\n{text_en}\n"
    )

    resp = client.responses.create(
        model=model,
        input=prompt,
        timeout=timeout_seconds,
    )
    out = (getattr(resp, "output_text", None) or "").strip()
    return out


def translate_en_to_ar_cached(text_en: str, *, db_path: str, timeout_seconds: float = 40.0) -> str:
    """
    Cache lookup:
      translations_cache(src_lang,tgt_lang,src_text,out_text) via storage.get_cached_translation / put_cached_translation

    Important behavior:
    - If DB is locked, still return a translation (skip caching instead of crashing).
    """
    text_en = (text_en or "").strip()
    if not text_en:
        return ""

    # Read cache (best-effort)
    try:
        hit = storage.get_cached_translation("en", "ar", text_en, db_path=db_path)
        if hit:
            return hit
    except sqlite3.OperationalError:
        # DB locked or table missing; ignore cache
        pass

    # Translate
    out = _openai_translate_en_to_ar(text_en, timeout_seconds=float(timeout_seconds))

    # Write cache (best-effort)
    try:
        storage.put_cached_translation("en", "ar", text_en, out, db_path=db_path)
    except sqlite3.OperationalError:
        # If locked, don't fail the tick
        pass

    return out