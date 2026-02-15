# drift_eval.py
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Tuple


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def update_confidence(
    conf_before: float,
    grounded_ratio: float,
    hallucinate_ratio: float,
    alpha_up: float,
    beta_down: float,
) -> float:
    """
    v1.6 logic:
    - confidence increases with grounded_ratio
    - decreases with hallucinate_ratio
    """
    return clamp01(conf_before + (alpha_up * grounded_ratio) - (beta_down * hallucinate_ratio))


# -----------------------------
# Local (no-OpenAI) text builders
# -----------------------------

_TONE_PREFIX: Dict[str, str] = {
    "human": "It lands close to the body: routine, fatigue, tenderness, and small decisions.",
    "liminal": "It feels suspended: thresholds, pauses, and the sense of waiting for a shape to form.",
    "environment": "It reads as atmosphere: conditions, currents, and signals carried by air and ground.",
    "environmental": "It reads as atmosphere: conditions, currents, and signals carried by air and ground.",
    "digital": "It arrives as pulses: fragments, latency, repetition, and sudden synchronization.",
    "infrastructure": "It shows as systems: routes, constraints, failures, repairs, and governance.",
    "more_than_human": "It stretches into longer cycles: migrations, species-scale pressure, deep rhythm.",
}


def heuristic_summary_line(mind_key: str) -> str:
    """
    One-line summary used for drift_memory.summary_text.
    """
    return _TONE_PREFIX.get(mind_key, "It adjusts across signals and gaps.")


def _clean_keep_paragraphs(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for ln in s.split("\n"):
        ln = " ".join((ln or "").split()).strip()
        lines.append(ln)
    out: List[str] = []
    empty_run = 0
    for ln in lines:
        if not ln:
            empty_run += 1
            if empty_run <= 1:
                out.append("")
        else:
            empty_run = 0
            out.append(ln)
    return "\n".join(out).strip()


def _normalize_titles(titles: List[str], k: int = 2) -> List[str]:
    out = []
    for t in (titles or []):
        t2 = " ".join((t or "").replace("\n", " ").split()).strip()
        if t2:
            out.append(t2)
    return out[:k]


def _target_word_count_from_axis(axis_text: str, min_words: int = 140, max_words: int = 260) -> int:
    words = [w for w in (" ".join((axis_text or "").split())).split(" ") if w]
    n = len(words)
    if n <= 0:
        return min_words
    return max(min_words, min(max_words, n))


def _trim_to_words_preserve_paragraphs(text: str, target_words: int) -> str:
    if not text:
        return ""
    parts = text.split("\n\n")
    kept: List[str] = []
    count = 0
    for p in parts:
        ws = [w for w in p.split() if w]
        if not ws:
            continue
        if count + len(ws) <= target_words:
            kept.append(" ".join(ws))
            count += len(ws)
        else:
            remaining = max(0, target_words - count)
            if remaining > 0:
                kept.append(" ".join(ws[:remaining]))
                count += remaining
            break
    out = "\n\n".join(kept).strip()
    if out and not out.endswith((".", "!", "?")):
        out += "."
    return out


def heuristic_drift_from_axis_parent(
    mind_key: str,
    *,
    axis_text: str,
    parent_text: str,
    event_titles: List[str],
    hallucination_level: float,
) -> str:
    """
    Local fallback drift body for drift_memory.drift_text.

    IMPORTANT:
    - NO mood-prefix at the top (that belongs in summary_text / keepsake_text)
    - Anchors length/structure to axis
    - Events appear only as subtle edge-pressure
    """
    axis = _clean_keep_paragraphs(axis_text or "")
    parent = _clean_keep_paragraphs(parent_text or "")

    # If previous drift was polluted with a mood-prefix, strip it out so drift_text can recover.
    mood = heuristic_summary_line(mind_key)
    if parent.startswith(mood):
        parent = parent[len(mood):].lstrip()

    if not axis:
        axis = parent
    if not parent:
        parent = axis
    if not axis and not parent:
        return "A memory axis remains present, even when details fade."

    target_words = _target_word_count_from_axis(axis, min_words=140, max_words=260)

    base = parent.strip()
    if len(base.split()) < int(target_words * 0.85):
        base = (base + "\n\n" + axis).strip()

    body = _trim_to_words_preserve_paragraphs(base, target_words)

    titles = _normalize_titles(event_titles, k=2)
    if titles:
        if hallucination_level >= 0.7:
            edge = "At the edge, distant signals accumulate as noise: " + " | ".join(titles) + "."
        elif hallucination_level > 0.0:
            edge = "At the edge, a few signals press lightly: " + " | ".join(titles) + "."
        else:
            edge = "At the edge, small signals appear: " + " | ".join(titles) + "."
        body = (body + "\n\n" + edge).strip()

    return body


def heuristic_keepsake_recap(
    mind_key: str,
    *,
    event_titles: List[str],
    hallucination_level: float,
) -> str:
    """
    Local fallback keepsake narration for drift_memory.keepsake_text.
    Mood-prefix is allowed here.
    """
    mood = heuristic_summary_line(mind_key)
    titles = _normalize_titles(event_titles, k=2)

    if not titles:
        return mood + " The feed is quiet, but the memory keeps drifting."

    stitched = " | ".join(titles)

    if hallucination_level >= 0.6:
        tail = "The signals do not fully align, so the keepsake trembles at its edges."
    elif hallucination_level >= 0.3:
        tail = "Some parts align, others blur; the keepsake keeps negotiating its confidence."
    else:
        tail = "The signals align cleanly enough to hold a stable contour."

    return f"{mood} {stitched}. {tail}"


def make_prompt_hash(params: Dict[str, Any], drift_text: str) -> str:
    """
    Stable hash used for drift reproducibility checks and auditing.
    """
    payload = json.dumps(params, ensure_ascii=False, sort_keys=True) + (drift_text or "")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()