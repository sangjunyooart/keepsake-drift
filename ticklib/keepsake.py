# ticklib/keepsake.py
from __future__ import annotations

import os
import re
from typing import List, Optional


def generate_keepsake_narration_openai(
    *,
    mind_key: str,
    original_en: str,
    prev_en: str,
    cur_en: str,
    invariants_phrases: List[str],
    evidence_fragments_en: List[str],
    model: Optional[str] = None,
) -> str:
    if not os.getenv("OPENAI_API_KEY"):
        return ""
    from openai import OpenAI

    original_en = (original_en or "").strip()[:900]
    prev_en = (prev_en or "").strip()[:900]
    cur_en = (cur_en or "").strip()[:900]

    inv = [re.sub(r"\s+", " ", (x or "").strip()) for x in (invariants_phrases or [])]
    inv = [x for x in inv if x][:10]

    frags = [f.strip() for f in (evidence_fragments_en or []) if f and f.strip()][:8]

    inv_block = "\n".join([f"- {x}" for x in inv]) if inv else "- (none)"
    frags_block = "\n".join([f"- {f[:220]}" for f in frags]) if frags else "- (none)"

    prompt = f"""
You are writing a short archive entry for a keepsake log in an artwork installation.

Mind lens: {mind_key}

Write exactly ONE paragraph in English, 40–80 words.
No headings, no bullet points, no labels.

Rules:
- Describe what shifted in THIS drift only — what is the main change from the previous state?
- Name the specific present-moment source that influenced this shift (the evidence fragments / world events).
- State how the temporality feels right now — its current emotional register.
- Third-person, observational. Honest, precise, slightly melancholic.
- Do NOT retell the full history or reference the origin. Each entry is one timestamped log entry in a scrollable archive.
- Never mention axis, prompt, model, instructions, or missing input.
- No profanity.
- Avoid religious or political claims, judgments, or provocations.
- Do not quote evidence fragments verbatim.

Previous drift:
{prev_en}

Current drift:
{cur_en}

Invariants that should stay recognizable:
{inv_block}

Present-moment signals (context only — name these as the source of the shift):
{frags_block}
""".strip()

    client = OpenAI()
    resp = client.responses.create(
        model=(model or os.getenv("OPENAI_MODEL", "gpt-5.2")),
        input=prompt,
        timeout=25.0,
    )
    out = (resp.output_text or "").strip()
    out = re.sub(r"\s+", " ", out).strip()
    return out[:800] if out else ""