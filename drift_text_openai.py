# drift_text_openai.py
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from openai import OpenAI

# Optional fallback translation if the model doesn't return Arabic
try:
    from ticklib.translate import translate_en_to_ar_cached  # type: ignore
except Exception:
    translate_en_to_ar_cached = None  # type: ignore

# ============================================================
# Tokenization MUST match your JS highlighter indices
#   EN: /([A-Za-z0-9]+(?:'[A-Za-z0-9]+)?)|([^A-Za-z0-9]+)/g
#   AR/Unicode: /([^\W_]+)|([\W_]+)/gu
# ============================================================

RE_EN_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
RE_U_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def _tok_en(text: str) -> List[str]:
    return RE_EN_WORD.findall(text or "")


def _tok_u(text: str) -> List[str]:
    return RE_U_WORD.findall(text or "")


def _build_token_ops(prev_tokens: List[str], cur_tokens: List[str]) -> Dict[str, Any]:
    sm = SequenceMatcher(a=prev_tokens, b=cur_tokens, autojunk=False)
    raw: List[Dict[str, Any]] = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag in ("insert", "replace"):
            raw.append(
                {
                    "op": tag,
                    "cur_span": [int(j1), int(j2)],
                    "cur_text": " ".join(cur_tokens[j1:j2]),
                }
            )
        # delete produces no underline in your current contract

    if not raw:
        return {"token_ops": []}

    # Merge contiguous / adjacent spans (including small gaps of 0 tokens)
    raw.sort(key=lambda x: (x["cur_span"][0], x["cur_span"][1]))
    merged: List[Dict[str, Any]] = [raw[0]]

    for item in raw[1:]:
        prev = merged[-1]
        a0, a1 = prev["cur_span"]
        b0, b1 = item["cur_span"]

        # If touching or overlapping, merge
        if b0 <= a1:
            prev["cur_span"] = [a0, max(a1, b1)]
            # Recompute cur_text for merged span
            prev["cur_text"] = " ".join(cur_tokens[prev["cur_span"][0] : prev["cur_span"][1]])
            prev["op"] = "replace"  # merged ops become replace
        else:
            merged.append(item)

    return {"token_ops": merged}


def build_en_delta(prev_en: str, cur_en: str) -> Dict[str, Any]:
    return _build_token_ops(_tok_en(prev_en or ""), _tok_en(cur_en or ""))


def build_ar_patch(prev_ar: str, cur_ar: str) -> Dict[str, Any]:
    return _build_token_ops(_tok_u(prev_ar or ""), _tok_u(cur_ar or ""))


def build_segment_aware_delta(
    drift_en: str,
    reps_en: List[Dict[str, str]],
    selected_segments: List[Dict[str, str]],
    drifted_keywords: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Build delta marking that highlights ONLY the replacement regions
    in the current drift text — the semantic drift triggered by RSS/event
    pressure — rather than diffing all tokens.

    If drifted_keywords is provided (from OpenAI), narrows marking to only
    those tokens within replacement regions whose text matches a keyword.

    Returns {"token_ops": [...]} on success, or None if any replacement
    text could not be located in drift_en (caller should fall back to
    build_en_delta).
    """
    drift_en = (drift_en or "").strip()
    if not reps_en or not drift_en:
        return {"token_ops": []}

    # Step 1: tokenize drift_en preserving character offsets
    tokens_with_offsets = [
        (m.group(), m.start(), m.end())
        for m in RE_EN_WORD.finditer(drift_en)
    ]
    if not tokens_with_offsets:
        return {"token_ops": []}

    # Step 2: build segment ordering for positional disambiguation
    seg_order = {}
    for i, seg in enumerate(selected_segments or []):
        sid = seg.get("segment_id", "")
        if sid:
            seg_order[sid] = i

    sorted_reps = sorted(
        reps_en,
        key=lambda r: seg_order.get(r.get("segment_id", ""), 999),
    )

    # Step 3: locate each replacement's character span in drift_en
    replacement_char_spans = []
    search_start = 0

    for rep in sorted_reps:
        rep_text = (rep.get("text") or "").strip()
        if not rep_text:
            continue

        idx = drift_en.find(rep_text, search_start)
        if idx < 0:
            # Try with normalized whitespace
            norm_rep = " ".join(rep_text.split())
            norm_drift = " ".join(drift_en.split())
            if norm_rep and norm_drift.find(norm_rep) >= 0:
                # Approximate: search in original with collapsed whitespace
                idx = drift_en.lower().find(norm_rep.lower(), search_start)
            if idx < 0:
                # Cannot locate replacement — signal fallback
                return None

        replacement_char_spans.append((idx, idx + len(rep_text)))
        search_start = idx + len(rep_text)

    if not replacement_char_spans:
        return {"token_ops": []}

    # Step 4: map character spans to token indices (replacement regions)
    replacement_token_indices = set()
    for (cs, ce) in replacement_char_spans:
        for i, (_word, tok_start, tok_end) in enumerate(tokens_with_offsets):
            if tok_start >= ce:
                break  # tokens are in order
            if tok_end > cs:
                replacement_token_indices.add(i)

    if not replacement_token_indices:
        return {"token_ops": []}

    # Step 5: keyword-level narrowing (if drifted_keywords provided)
    # Only mark tokens within replacement regions that match a keyword
    kw_list = [kw.strip().lower() for kw in (drifted_keywords or []) if (kw or "").strip()]
    if kw_list:
        # Build keyword token sets: tokenize each keyword phrase and match
        # multi-word keywords as consecutive token runs
        marked_token_indices = set()
        for kw in kw_list:
            kw_tokens = [w.lower() for w in RE_EN_WORD.findall(kw) if w]
            if not kw_tokens:
                continue
            kw_len = len(kw_tokens)
            # Scan through all tokens in replacement regions for this keyword
            rep_sorted = sorted(replacement_token_indices)
            for start_idx in rep_sorted:
                if start_idx + kw_len - 1 > max(rep_sorted):
                    break
                match = True
                for k, kw_tok in enumerate(kw_tokens):
                    ti = start_idx + k
                    if ti >= len(tokens_with_offsets):
                        match = False
                        break
                    if tokens_with_offsets[ti][0].lower() != kw_tok:
                        match = False
                        break
                if match:
                    for k in range(kw_len):
                        marked_token_indices.add(start_idx + k)
        # Fallback: if no keyword tokens matched, mark all replacement tokens
        if not marked_token_indices:
            marked_token_indices = replacement_token_indices
    else:
        marked_token_indices = replacement_token_indices

    if not marked_token_indices:
        return {"token_ops": []}

    # Step 6: group consecutive indices into contiguous spans
    sorted_indices = sorted(marked_token_indices)
    spans = []
    span_start = sorted_indices[0]
    span_end = sorted_indices[0] + 1

    for idx in sorted_indices[1:]:
        if idx == span_end:
            span_end = idx + 1
        else:
            spans.append((span_start, span_end))
            span_start = idx
            span_end = idx + 1
    spans.append((span_start, span_end))

    # Step 7: emit token_ops
    token_ops = []
    for (a, b) in spans:
        token_ops.append({
            "op": "replace",
            "cur_span": [a, b],
            "cur_text": " ".join(t[0] for t in tokens_with_offsets[a:b]),
        })

    return {"token_ops": token_ops}


def build_ar_patch_from_en_delta(
    en_delta: Dict[str, Any],
    drift_en: str,
    drift_ar: str,
) -> Dict[str, Any]:
    """
    Map EN delta token_ops → AR token_ops by proportional word-index scaling.

    Since EN and AR are translations of the same content, word positions
    are roughly proportional. This avoids the problem of diffing two
    independent Arabic translations (which produces noise or nothing).
    """
    ops = en_delta.get("token_ops") or []
    if not ops:
        return {"token_ops": []}

    en_tokens = _tok_en(drift_en or "")
    ar_tokens = _tok_u(drift_ar or "")

    en_count = len(en_tokens)
    ar_count = len(ar_tokens)

    if en_count == 0 or ar_count == 0:
        return {"token_ops": []}

    ratio = ar_count / en_count

    ar_ops: List[Dict[str, Any]] = []
    for op in ops:
        span = op.get("cur_span")
        if not span or len(span) < 2:
            continue

        en_a, en_b = int(span[0]), int(span[1])

        # Scale to AR indices, clamp to valid range
        ar_a = max(0, min(int(round(en_a * ratio)), ar_count - 1))
        ar_b = max(ar_a + 1, min(int(round(en_b * ratio)), ar_count))

        ar_ops.append({
            "op": op.get("op", "replace"),
            "cur_span": [ar_a, ar_b],
            "cur_text": " ".join(ar_tokens[ar_a:ar_b]),
        })

    # Merge overlapping/adjacent spans
    if len(ar_ops) > 1:
        ar_ops.sort(key=lambda x: (x["cur_span"][0], x["cur_span"][1]))
        merged = [ar_ops[0]]
        for item in ar_ops[1:]:
            prev = merged[-1]
            a0, a1 = prev["cur_span"]
            b0, b1 = item["cur_span"]
            if b0 <= a1:
                prev["cur_span"] = [a0, max(a1, b1)]
                prev["cur_text"] = " ".join(ar_tokens[prev["cur_span"][0]:prev["cur_span"][1]])
                prev["op"] = "replace"
            else:
                merged.append(item)
        ar_ops = merged

    return {"token_ops": ar_ops}


def extract_invariables_openai(
    memories: Dict[str, str],
    *,
    model: Optional[str] = None,
    timeout_seconds: float = 45.0,
) -> Dict[str, List[Dict[str, str]]]:
    """
    Extract word/phrase-level invariables from original memory texts via OpenAI.

    One bundled call for all minds. Returns categorized invariables per mind_key:
      - proper_noun: place names, person names, titles of works, geo-specific terms
      - temporal: dates, times, seasons, durations, time-of-day markers
      - sensory: sensory impressions, material textures, animal/nature names, emotional/atmospheric cues

    Args:
        memories: {mind_key: original_text_en}
    Returns:
        {mind_key: [{"phrase": "...", "category": "...", "id": "..."}]}
    """
    todo = {k: v.strip() for k, v in memories.items() if (v or "").strip()}
    if not todo:
        return {}

    model = model or os.getenv("OPENAI_TEXT_MODEL", os.getenv("OPENAI_DRIFT_MODEL", "gpt-4.1-mini"))

    input_json = json.dumps(todo, ensure_ascii=False, indent=2)

    prompt = f"""You are analyzing original memory texts to extract invariable words and phrases.

For each memory text, extract ONLY specific words or short phrases (NOT full sentences) that anchor the memory to time, space, and sensory experience.

Categories:
- "proper_noun": proper nouns, place names, person names, titles of works, geographic terms, cultural references
- "temporal": dates, times, seasons, durations, time-of-day markers, years
- "sensory": sensory impressions (sounds, sights, textures), animal/nature names, emotional/atmospheric cues, material details

Rules:
- Extract individual words or short phrases (1-5 words typically), NEVER full sentences.
- Each phrase must appear verbatim in the original text.
- Be thorough — capture all meaningful anchors, typically 8-15 per text.
- Proper nouns include titles of musical pieces, place names, cultural terms.
- Sensory includes specific creatures, natural phenomena, and subjective impressions.

Return ONLY strict JSON:
{{
  "<mind_key>": [
    {{"phrase": "...", "category": "proper_noun|temporal|sensory"}}
  ],
  ...
}}

Input:
{input_json}
"""

    result = _openai_json(prompt, model=model, timeout_seconds=timeout_seconds)

    out: Dict[str, List[Dict[str, str]]] = {}
    if isinstance(result, dict):
        for mk in todo:
            items = result.get(mk)
            if not isinstance(items, list):
                continue
            inv_list: List[Dict[str, str]] = []
            prefix = mk[:2]
            for i, item in enumerate(items, start=1):
                if not isinstance(item, dict):
                    continue
                phrase = str(item.get("phrase") or "").strip()
                category = str(item.get("category") or "").strip()
                if not phrase:
                    continue
                if category not in ("proper_noun", "temporal", "sensory"):
                    category = "sensory"  # default fallback
                inv_list.append({
                    "phrase": phrase,
                    "category": category,
                    "id": f"{prefix}_inv_{i}",
                })
            out[mk] = inv_list

    return out


def translate_bundled_en_to_ar(
    texts: Dict[str, str],
    *,
    model: Optional[str] = None,
    timeout_seconds: float = 45.0,
    db_path: Optional[str] = None,
    target_lang: Optional[str] = None,
) -> Dict[str, str]:
    """
    Translate multiple English texts to the active second language in a single OpenAI call.
    Checks translation cache first, only sends uncached texts to OpenAI.

    Args:
        texts: mapping of {key: english_text} — keys are opaque identifiers
               (e.g. "human__drift", "liminal__summary")
        target_lang: language code ("ar", "el", "pt-br"). Defaults to config.SECOND_LANG.
    Returns:
        mapping of {key: translated_text}
    """
    import config as _cfg
    tl = target_lang or _cfg.SECOND_LANG
    lc = _cfg.LANG_CONFIG.get(tl, _cfg.LANG_CONFIG["ar"])
    lang_name = lc["name"]
    lang_rules = lc["rules"]

    # Filter out empty values
    todo = {k: v.strip() for k, v in texts.items() if (v or "").strip()}
    if not todo:
        return {}

    # Check translation cache first
    cached: Dict[str, str] = {}
    remaining: Dict[str, str] = {}
    if db_path:
        try:
            import storage as _st
            for k, en_text in todo.items():
                try:
                    hit = _st.get_cached_translation("en", tl, en_text, db_path=db_path)
                    if hit:
                        cached[k] = hit
                        continue
                except Exception:
                    pass
                remaining[k] = en_text
        except Exception:
            remaining = dict(todo)
    else:
        remaining = dict(todo)

    if not remaining:
        return cached

    model = model or os.getenv("OPENAI_TRANSLATE_MODEL", os.getenv("OPENAI_DRIFT_MODEL", "gpt-4.1-mini"))

    # Build bundled prompt using proper JSON serialization to avoid
    # quoting issues when texts contain double-quotes or special chars
    input_json = json.dumps(remaining, ensure_ascii=False, indent=2)

    prompt = f"""Translate each English text below into {lang_name}.
Return ONLY strict JSON: a single object mapping each key to its {lang_name} translation.
Use the exact same keys as the input.

Rules:
- Output {lang_name} only for each value. CRITICAL: ALL English words must be translated to {lang_name} — do not leave any Latin script words untranslated (including technical terms, loanwords, or colloquialisms).
- {lang_rules}
- Preserve paragraph breaks (\\n) within each text.

Input:
{input_json}
"""

    try:
        result = _openai_json(prompt, model=model, timeout_seconds=timeout_seconds)
    except Exception:
        # Fallback: return whatever we cached
        return cached

    out = dict(cached)
    if isinstance(result, dict):
        for k in remaining:
            ar_text = str(result.get(k) or "").strip()
            if ar_text:
                out[k] = ar_text
                # Cache individual translations
                if db_path:
                    try:
                        import storage as _st
                        _st.put_cached_translation("en", tl, remaining[k], ar_text, db_path=db_path)
                    except Exception:
                        pass

    return out


def translate_drift_pairs_en_to_ar(
    pairs: Dict[str, Dict[str, str]],
    *,
    model: Optional[str] = None,
    timeout_seconds: float = 55.0,
    db_path: Optional[str] = None,
    target_lang: Optional[str] = None,
) -> Dict[str, Dict[str, str]]:
    """
    Translate (prev_en, drift_en) pairs to the active second language such that
    unchanged English sentences produce identical L2, and only actually-drifted
    segments differ between prev_ar and drift_ar.

    Args:
        pairs: {mind_key: {"prev_en": "...", "drift_en": "..."}}
        target_lang: language code ("ar", "el", "pt-br"). Defaults to config.SECOND_LANG.
    Returns:
        {mind_key: {"prev_ar": "...", "drift_ar": "..."}}
    """
    import config as _cfg
    tl = target_lang or _cfg.SECOND_LANG
    lc = _cfg.LANG_CONFIG.get(tl, _cfg.LANG_CONFIG["ar"])
    lang_name = lc["name"]
    lang_rules = lc["rules"]

    if not pairs:
        return {}

    model = model or os.getenv("OPENAI_TRANSLATE_MODEL", os.getenv("OPENAI_DRIFT_MODEL", "gpt-4.1-mini"))

    # Build input: for each mind, send both prev and drift
    input_obj: Dict[str, Dict[str, str]] = {}
    for mk, p in pairs.items():
        prev = (p.get("prev_en") or "").strip()
        cur = (p.get("drift_en") or "").strip()
        if cur:
            input_obj[mk] = {"prev_en": prev, "drift_en": cur}

    if not input_obj:
        return {}

    # CHUNKING: To avoid hitting token limits, translate in batches of 2 minds at a time
    mind_keys = list(input_obj.keys())
    chunk_size = 2
    all_results: Dict[str, Dict[str, str]] = {}

    for i in range(0, len(mind_keys), chunk_size):
        chunk_keys = mind_keys[i:i+chunk_size]
        chunk_obj = {k: input_obj[k] for k in chunk_keys}

        input_json = json.dumps(chunk_obj, ensure_ascii=False, indent=2)

        prompt = f"""You are translating paired English texts (previous and current versions) into {lang_name}.

CRITICAL RULE: Sentences that are IDENTICAL between "prev_en" and "drift_en" MUST have the EXACT SAME {lang_name} translation in both "prev_ar" and "drift_ar". Only sentences that actually changed in English should have different {lang_name} wording.

Return ONLY strict JSON with this structure:
{{
  "<mind_key>": {{
    "prev_ar": "...",
    "drift_ar": "..."
  }},
  ...
}}

CRITICAL JSON FORMATTING RULES:
- All string values must be properly escaped (use \\n for newlines, \\" for quotes)
- NO trailing commas before closing braces or brackets
- NO unterminated strings - every quote must be closed
- Ensure all Unicode/Arabic characters are preserved exactly
- Double-check that all braces and brackets are balanced

Rules:
- Output {lang_name} only for each value. CRITICAL: ALL English words must be translated to {lang_name} — do not leave any Latin script words untranslated (including technical terms, loanwords, or colloquialisms).
- {lang_rules}
- Identical English sentences → identical {lang_name} output in both prev_ar and drift_ar.

Input:
{input_json}
"""

        try:
            result = _openai_json(prompt, model=model, timeout_seconds=timeout_seconds)
        except Exception as e:
            import sys
            import traceback
            print(f"[ERROR translate_drift_pairs_en_to_ar chunk {i//chunk_size + 1}] {type(e).__name__}: {str(e)}", file=sys.stderr)
            print(f"[ERROR translate_drift_pairs_en_to_ar chunk {i//chunk_size + 1}] {traceback.format_exc()}", file=sys.stderr)
            continue

        if isinstance(result, dict):
            for mk in chunk_keys:
                entry = result.get(mk)
                if isinstance(entry, dict):
                    prev_ar = str(entry.get("prev_ar") or "").strip()
                    drift_ar = str(entry.get("drift_ar") or "").strip()
                    if drift_ar:
                        all_results[mk] = {"prev_ar": prev_ar, "drift_ar": drift_ar}

                        # Cache individual translations
                        if db_path:
                            try:
                                import storage as _st
                                if prev_ar and chunk_obj[mk]["prev_en"]:
                                    _st.put_cached_translation("en", tl, chunk_obj[mk]["prev_en"], prev_ar, db_path=db_path)
                                _st.put_cached_translation("en", tl, chunk_obj[mk]["drift_en"], drift_ar, db_path=db_path)
                            except Exception:
                                pass

    return all_results


def _sha256(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def _client() -> OpenAI:
    return OpenAI()


def _openai_json(prompt: str, model: str, timeout_seconds: float) -> Dict[str, Any]:
    client = _client()
    resp = client.responses.create(
        model=model,
        input=prompt,
        timeout=timeout_seconds,
    )

    text = None
    try:
        text = resp.output_text
    except Exception:
        pass

    if not text:
        try:
            parts = []
            for item in (resp.output or []):
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", None) in ("output_text", "text"):
                        parts.append(getattr(c, "text", ""))
            text = "\n".join(parts).strip()
        except Exception:
            text = None

    if not text:
        raise RuntimeError("OpenAI response had no text output to parse as JSON.")

    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()

    try:
        return json.loads(t)
    except json.JSONDecodeError as e:
        # Try to fix common JSON issues from OpenAI (trailing commas, unterminated strings, etc.)
        try:
            fixed = t
            # Remove trailing commas before closing braces/brackets
            fixed = re.sub(r',\s*}', '}', fixed)
            fixed = re.sub(r',\s*]', ']', fixed)

            # Try parsing the repaired JSON
            return json.loads(fixed)
        except json.JSONDecodeError as repair_err:
            # If still failing, try to salvage what we can by using Python's ast.literal_eval
            # or just fail with detailed error
            import sys
            print(f"[ERROR _openai_json] JSON parse failed. Error: {str(repair_err)}", file=sys.stderr)
            print(f"[ERROR _openai_json] Full response text ({len(t)} chars):", file=sys.stderr)
            print(t, file=sys.stderr)
            raise RuntimeError(f"Failed to parse model JSON even after repair. Error: {str(repair_err)}\nFirst 500 chars:\n{t[:500]}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to parse model JSON. First 300 chars:\n{t[:300]}") from e


@dataclass
class MindCtx:
    mind_id: int
    mind_key: str
    axis_en: str
    prev_en: str
    prev_marked_en: str
    sensory_anchors: List[str]
    selected_segments: List[Dict[str, str]]
    invariables: List[Dict[str, str]]  # word/phrase-level invariables
    prev_ar: str
    prev_version: int
    prev_drift_id: Optional[int]
    event_titles: Optional[List[str]] = None  # RSS event titles for justification
    drift_direction: str = ""  # Stage 1 lens interpretation (how this mind reads headlines)
    infiltrating_imagery: Optional[List[str]] = None  # Concrete phrases from headlines to absorb
    resonance: float = 0.5  # 0.0-1.0 how well RSS headlines resonate with this mind's lens


def generate_lens_interpretations(
    *,
    event_titles: List[str],
    minds: List[Dict[str, Any]],
    model: Optional[str] = None,
    timeout_seconds: float = 30.0,
) -> Dict[str, Any]:
    """
    Stage 1: Each temporality interprets RSS headlines through its lens.

    Returns {mind_key: {drift_direction: str, infiltrating_imagery: [str]}} —
    a short (2-3 sentence) interpretation + concrete phrases from headlines
    that should seep into this mind's memory.

    Args:
        event_titles: list of RSS headline strings
        minds: list of dicts with keys: mind_key, perspective, decay_policy
            - perspective: from PROTOTYPE_SEED_TEXT (e.g. "Everyday human time...")
            - decay_policy: from DECAY_POLICIES (dict with hard/soft/volatile)
        model: OpenAI model name
        timeout_seconds: API timeout
    """
    if not event_titles or not minds:
        return {}

    model = model or os.getenv("OPENAI_TEXT_MODEL", os.getenv("OPENAI_DRIFT_MODEL", "gpt-4.1-mini"))

    # Build per-mind blocks — each mind gets its OWN curated headlines
    blocks: List[str] = []
    for m in minds:
        mk = str(m.get("mind_key") or "")
        perspective = str(m.get("perspective") or mk)
        policy = m.get("decay_policy") or {}

        preserves = ", ".join(policy.get("hard_invariants", []))
        lets_blur = ", ".join(policy.get("volatile_details", []))

        # Per-mind curated headlines (pre-filtered by resonance)
        mind_headlines = m.get("curated_headlines") or event_titles[:4]
        hl_lines = "\n".join([f'    - "{t}"' for t in mind_headlines[:5]])

        mind_resonance_val = float(m.get("resonance", 0.5))
        resonance_note = ""
        if mind_resonance_val < 0.3:
            resonance_note = (
                f"RESONANCE: {mind_resonance_val:.2f} — LOW. These headlines barely connect to this mind's lens. "
                f"The drift_direction should reflect disorientation — this mind struggles to find purchase "
                f"in the present moment. The infiltrating_imagery should feel untethered, disconnected.\n"
            )
        elif mind_resonance_val < 0.6:
            resonance_note = f"RESONANCE: {mind_resonance_val:.2f} — MODERATE. Partial connection.\n"

        blocks.append(
            f"[mind:{mk}]\n"
            f"PERSPECTIVE: {perspective}\n"
            f"PRESERVES: {preserves}\n"
            f"LETS BLUR: {lets_blur}\n"
            f"{resonance_note}"
            f"HEADLINES THIS MIND RESONATES WITH:\n{hl_lines}"
        )

    prompt = f"""You are interpreting current world events through the lens of six temporality-minds.
Each mind has a unique perspective on time, memory, and experience.

IMPORTANT: Each mind below has been given DIFFERENT headlines — the ones most resonant
with its particular lens. Respond to ONLY the headlines listed for that specific mind.

For EACH mind, explain WHAT it notices in its headlines,
HOW it wants to shift (drift) its memory, and what EMOTIONAL TEXTURES the headlines evoke.

Rules:
- Each drift_direction should be specific, not generic. Reference actual headline content.
- Each mind responds ONLY to its own curated headlines, not all headlines.
- Focus on conceptual direction: what should blur, what should sharpen, what new resonance enters.
- Do NOT mention "the human mind" or "the infrastructure mind" — write AS the mind's inner voice.
- CRITICAL: Include 2-4 SENSORIAL or EMOTIONAL TEXTURES that the headlines evoke for this mind.
  These are NOT literal headline words — they are the atmospheric qualities a memory takes on
  when recalled under the pressure of the present. For example, if a headline mentions "wildfire smoke
  drifts across the coast", the environment mind might feel "dry stillness", "haze-thickened air",
  "something burning far away". The human mind might feel "the weight of a warm afternoon",
  "a taste that wasn't there before". These are moods, not news.

Return ONLY strict JSON:
{{
  "minds": {{
    "<mind_key>": {{
      "drift_direction": "2-3 sentences of interpretive direction...",
      "infiltrating_imagery": ["concrete phrase 1", "concrete phrase 2", "concrete phrase 3"]
    }},
    ...
  }}
}}

{chr(10).join(blocks)}
""".strip()

    try:
        result = _openai_json(prompt, model=model, timeout_seconds=timeout_seconds)
    except Exception:
        return {}

    out: Dict[str, Any] = {}
    if isinstance(result, dict):
        minds_obj = result.get("minds") or result  # fallback if no "minds" wrapper
        if isinstance(minds_obj, dict):
            for m in minds:
                mk = str(m.get("mind_key") or "")
                entry = minds_obj.get(mk)
                if isinstance(entry, dict):
                    direction = str(entry.get("drift_direction") or "").strip()
                    imagery = entry.get("infiltrating_imagery") or []
                    if not isinstance(imagery, list):
                        imagery = []
                    imagery = [str(x).strip() for x in imagery if str(x).strip()]
                    if direction:
                        out[mk] = {
                            "drift_direction": direction,
                            "infiltrating_imagery": imagery,
                        }
                elif isinstance(entry, str):
                    out[mk] = {
                        "drift_direction": entry.strip(),
                        "infiltrating_imagery": [],
                    }

    return out


def _build_prompt(tick_id: int, event_ids: List[int], minds: List[MindCtx]) -> str:
    """
    Single-call, patch-based drift contract.

    The model MUST:
    - Only rewrite the provided selected segments (by segment_id).
    - Provide EN and L2 replacements for each segment_id.
    - Provide one-sentence recap in EN and L2.
    - Preserve invariable words/phrases (they can be rephrased at lower resolution
      but their semantic identity must persist).
    """
    import config as _cfg
    _lc = _cfg.LANG_CONFIG.get(_cfg.SECOND_LANG, _cfg.LANG_CONFIG["ar"])
    _l2_name = _lc["name"]
    _l2_rules = _lc["rules"]

    header = f"""
You are a memory-drift engine for six temporalities.

IMPORTANT: All fields ending in "_ar" must contain {_l2_name} text (not necessarily Arabic).
{_l2_name} rules: {_l2_rules}

Return ONLY strict JSON with this structure:
{{
  "minds": {{
    "<mind_key>": {{
      "replacements_en": [{{"segment_id":"...","text":"..."}} ...],
      "replacements_ar": [{{"segment_id":"...","text":"..."}} ...],
      "drifted_keywords": ["word1", "phrase two", ...],
      "justification_en": "...",
      "justification_ar": "...",
      "keepsake_en": "...",
      "keepsake_ar": "...",
      "summary_en": "...",
      "summary_ar": "..."
    }},
    ...
  }}
}}

Rules (non-negotiable):
- Do NOT rewrite the whole text. Only rewrite the selected segments listed for that mind.
- Each replacement must correspond to an input segment_id.
- Keep the same approximate length for each segment (do not expand into paragraphs).
- The drift operates at TWO LEVELS:
  A) INVARIABLES (emotional anchors — these SURVIVE):
    - [sensory] invariables are UNTOUCHABLE. They must appear in the text using their EXACT original words. These are the emotional anchors that persist across all drifts — the specific quality of light, the texture, the sound. They are what makes the memory THIS memory. NEVER rephrase, blur, or replace them.
    - [proper_noun] invariables can lose specificity over time (e.g., "Nebraska City" -> "a small Nebraska town") but must remain recognizable as the same referent.
    - [temporal] invariables can lose precision (e.g., "11:00 a.m." -> "late morning") but must remain present.
    - If an invariable phrase appears in a segment you are rewriting, it MUST survive in the output. Build the new sentence AROUND the invariable, not instead of it.
  B) SEMANTIC DRIFT (shaking meaning — NOT rephrasing):
    - Drift is NOT paraphrasing. Drift is NOT finding synonyms or rewriting sentences in different words.
    - Drift means the MEANING itself shifts slightly. The same words might stay, but what they point to trembles. A "quiet garden" doesn't become "a still garden" (that's rephrasing) — it becomes "a quiet garden" where the quiet now carries unease, or anticipation, or loss. The semantic ground under the words moves.
    - The RSS headlines create pressure on the meaning. The memory doesn't absorb headline vocabulary — the headlines pressure what the existing words MEAN. Think of it as: recalling the same sentence on a day when wildfires are in the news — "the garden was quiet that afternoon" still says "quiet," but the quiet now means something drier, more fragile.
    - The DRIFT DIRECTION and INFILTRATING IMAGERY guide the emotional register and sensorial quality. Use them as atmospheric pressure on MEANING, not as vocabulary to swap in.
    - Each rewrite should feel like the SAME memory where the meaning has shifted underfoot — not a rephrased version of the same memory.
  C) HALLUCINATION (semantic instability — NOT visual fog):
    - Hallucination occurs when the temporality's lens does not resonate with the ingested present-moment signals. The memory becomes semantically unstable — words start pointing to things that don't quite connect to the original emotional core.
    - This is NOT about making text dreamy or foggy. It is about the meaning becoming unreliable. A hallucinating memory might say something that sounds coherent but no longer truly connects to what was felt. The structure holds but the semantic ground has shifted away from the anchor.
- No lists, no headings, no quotes, no markdown.
- drifted_keywords: list ONLY the specific words or short phrases (1-4 words each) in the replacement text that represent the SEMANTIC drift — the words whose meaning or emotional quality changed, not mere spelling variants. Typically 3-8 keywords per mind.
- justification_en/ar: 1-3 sentences explaining WHY these specific segments drifted. Reference the temporality's unique perspective (its lens) and how the RSS event context pressured the drift. Write in present tense.
- keepsake_en/ar: 1-2 SHORT sentences. First person — spoken by the temporality itself ("I") about what it just felt shift.
  Rules:
  - Write as "I" — the perceiver noting what changed in its own sensing. Not about "the memory" as an object.
  - Name one concrete shift (sensory, emotional, spatial) and what present-moment signal caused it.
  - NEVER start with "The memory", "Memory holds", "Memory feels", "My memory", or any form of "memory" as subject.
  - NEVER use the same sentence opening across different minds. Each entry must start differently.
  - Keep it under 30 words. Brief, direct, like a field note written mid-drift.
Tick: {tick_id}
Event IDs (may be empty): {event_ids}
""".strip()

    blocks: List[str] = []
    for m in minds:
        segs = m.selected_segments or []
        seg_lines = "\n".join([f"- {s['segment_id']}: {s['text']}" for s in segs]) if segs else "- (none)"

        # Build invariables section (word/phrase level)
        invs = m.invariables or []
        if invs:
            inv_lines = "\n".join([
                f'  - [{inv.get("category", "sensory")}] "{inv.get("phrase", "")}"'
                for inv in invs if inv.get("phrase")
            ])
        else:
            # Fallback to legacy sensory anchors
            anchors = [a for a in (m.sensory_anchors or []) if (a or "").strip()]
            inv_lines = "\n".join([f"  - [sensory] \"{a}\"" for a in anchors]) if anchors else "  (none)"

        # Event titles context for justification
        titles = m.event_titles or []
        if titles:
            evt_lines = "\n".join([f"  - \"{t}\"" for t in titles[:5]])
        else:
            evt_lines = "  (no events this tick)"

        # Drift direction from Stage 1 lens interpretation
        direction = (m.drift_direction or "").strip()
        if direction:
            direction_block = (
                f"\nDRIFT DIRECTION (this mind's interpretation of headlines — follow this to guide your rewrites):\n"
                f"{direction}\n"
            )
        else:
            direction_block = ""

        # Atmospheric imagery from Stage 1 — emotional textures to absorb (NOT literal vocabulary)
        imagery = m.infiltrating_imagery or []
        if imagery:
            imagery_block = (
                f"\nATMOSPHERIC IMAGERY (emotional textures drawn from headlines — use these to COLOR the "
                f"mood and sensorial quality of your rewrites, NOT as literal words to insert):\n"
                + "\n".join([f'  - "{img}"' for img in imagery[:6]])
                + "\n"
            )
        else:
            imagery_block = ""

        # Resonance-based hallucination pressure
        resonance_val = getattr(m, "resonance", 0.5)
        if resonance_val < 0.3:
            resonance_block = (
                f"\nRESONANCE: {resonance_val:.2f} — LOW. "
                f"The present moment does NOT connect to this temporality's lens. "
                f"The memory should become SEMANTICALLY UNSTABLE in this drift. "
                f"This means: the words can stay similar, but what they point to should feel "
                f"disconnected from the original emotional core. Sentences should sound coherent "
                f"but carry a meaning that has slipped sideways — like remembering something "
                f"accurately but no longer feeling why it mattered. The invariables still survive "
                f"as exact words, but the sentences around them lose their emotional grounding. "
                f"This is hallucination: the structure holds but the semantic floor has moved.\n"
            )
        elif resonance_val < 0.6:
            resonance_block = (
                f"\nRESONANCE: {resonance_val:.2f} — MODERATE. "
                f"Some headlines connect to this lens, others don't. "
                f"The drift should carry mild instability — meaning shifts but stays "
                f"oriented toward the original emotional core.\n"
            )
        else:
            resonance_block = (
                f"\nRESONANCE: {resonance_val:.2f} — HIGH. "
                f"The present moment resonates strongly with this temporality's lens. "
                f"The drift should feel grounded — meaning shifts clearly along the "
                f"pressure from headlines, staying connected to the emotional core.\n"
            )

        blocks.append(
            f"""
[mind:{m.mind_key}]
DRIFT: {m.prev_version + 1} (this is drift {m.prev_version + 1}; the AXIS below is the original memory, drift 0)

AXIS (the ORIGINAL memory — drift 0 — the emotional anchor everything drifts from):
{(m.axis_en or "").strip()}

INVARIABLES (emotional anchors — [sensory] ones must survive VERBATIM as exact words; [proper_noun] and [temporal] can lose resolution but must remain recognizable):
{inv_lines}

EVENT CONTEXT (RSS headlines this mind resonates with):
{evt_lines}
{resonance_block}{direction_block}{imagery_block}
PREV (drift {m.prev_version} — the current state of this memory before this tick's drift):
{(m.prev_en or "").strip()}

SELECTED SEGMENTS (ONLY these can be rewritten; keep 1:1 segment count):
{seg_lines}
""".strip()
        )

    return header + "\n\n" + "\n\n".join(blocks)


def _safe_translate_en_to_ar(text_en: str, db_path: str) -> str:
    if not text_en:
        return ""
    if translate_en_to_ar_cached is None:
        return ""
    try:
        return translate_en_to_ar_cached(text_en, db_path=db_path)  # type: ignore
    except Exception:
        return ""


def generate_bundled_drifts_openai(
    *,
    db_path: str,
    tick_id: int,
    event_ids: Optional[List[int]] = None,
    event_titles: Optional[List[str]] = None,
    drift_directions: Optional[Dict[str, Any]] = None,
    minds: List[Dict[str, Any]],
    model: Optional[str] = None,
    timeout_seconds: float = 120.0,
) -> Dict[str, Any]:
    event_ids = event_ids or []
    drift_directions = drift_directions or {}
    model = model or os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")

    mind_ctxs: List[MindCtx] = []
    for m in minds:
        mk = str(m["mind_key"])
        # drift_directions can be Dict[str, str] (legacy) or Dict[str, dict] (new with imagery)
        dd_entry = drift_directions.get(mk) or ""
        if isinstance(dd_entry, dict):
            dd_text = str(dd_entry.get("drift_direction") or "")
            dd_imagery = list(dd_entry.get("infiltrating_imagery") or [])
        else:
            dd_text = str(dd_entry)
            dd_imagery = []
        mind_ctxs.append(
            MindCtx(
                mind_id=int(m["mind_id"]),
                mind_key=mk,
                axis_en=str(m.get("axis_en") or ""),
                prev_en=str(m.get("prev_en") or ""),
                prev_marked_en=str(m.get("prev_marked_en") or ""),
                sensory_anchors=list(m.get("sensory_anchors") or []),
                selected_segments=list(m.get("selected_segments") or []),
                invariables=list(m.get("invariables") or []),
                prev_ar=str(m.get("prev_ar") or ""),
                prev_version=int(m.get("prev_version") or 0),
                prev_drift_id=(int(m["prev_drift_id"]) if m.get("prev_drift_id") is not None else None),
                event_titles=list(m.get("event_titles") or event_titles or []),
                drift_direction=dd_text,
                infiltrating_imagery=dd_imagery,
                resonance=float(m.get("resonance", 0.5)),
            )
        )

    prompt = _build_prompt(int(tick_id), list(event_ids), mind_ctxs)
    prompt_hash = _sha256(prompt)

    j = _openai_json(prompt, model=model, timeout_seconds=float(timeout_seconds))
    out_minds = (j.get("minds") or {}) if isinstance(j, dict) else {}
    if not isinstance(out_minds, dict) or not out_minds:
        raise RuntimeError("Model returned JSON but missing minds object.")

    bundled: Dict[str, Any] = {"tick_id": int(tick_id), "prompt_hash": prompt_hash, "minds": {}}

    # IMPORTANT:
    # We compute token deltas AFTER pipeline reconstruction (pipeline rebuilds drift text),
    # but we still return delta_json placeholders here; pipeline will overwrite if needed.
    # To keep your existing structure, we compute deltas here using prev_en + "best effort"
    # by applying segment replacements onto prev_en *without markers* (approx).
    #
    # Pipeline will compute deltas from reconstructed drift_text anyway (preferred).
    #
    # In this implementation, we compute deltas in pipeline; here we return empty deltas,
    # and pipeline fills them by calling build_en_delta/build_ar_patch again if desired.
    #
    # To stay compatible with your current pipeline expectations, we *do* compute here
    # using a simple substitution based on exact segment text where possible.

    def _apply_simple(prev: str, reps: List[Dict[str, str]], segs: List[Dict[str, str]]) -> str:
        txt = prev or ""
        seg_map = {s["segment_id"]: s["text"] for s in segs if s.get("segment_id") and s.get("text")}
        for r in reps:
            sid = (r.get("segment_id") or "").strip()
            newt = (r.get("text") or "").strip()
            oldt = seg_map.get(sid)
            if sid and oldt and oldt in txt:
                txt = txt.replace(oldt, newt, 1)
        return " ".join(txt.split()).strip()

    for m in mind_ctxs:
        payload = out_minds.get(m.mind_key) or {}
        reps_en = payload.get("replacements_en") or []
        reps_ar = payload.get("replacements_ar") or []
        if not isinstance(reps_en, list):
            reps_en = []
        if not isinstance(reps_ar, list):
            reps_ar = []

        # If model omitted replacements, keep empty list (pipeline will keep prev text)
        summary_en = str(payload.get("summary_en") or "").strip()
        summary_ar = str(payload.get("summary_ar") or "").strip()

        # NOTE: Do NOT fallback-translate summaries here individually.
        # Pipeline Phase 2 handles all missing Arabic in one bundled call.

        # Best-effort for deltas (pipeline reconstruction is authoritative)
        approx_en = _apply_simple(m.prev_en, reps_en, m.selected_segments)
        en_delta_obj = build_en_delta(m.prev_en or "", approx_en)

        approx_ar = ""
        if reps_ar:
            approx_ar = _apply_simple(m.prev_en, reps_ar, m.selected_segments)
        # If we don't have prev_ar or approx_ar, patch is empty
        ar_patch_obj = build_ar_patch(m.prev_ar or "", approx_ar) if (m.prev_ar and approx_ar) else {"token_ops": []}

        # Parse new fields: drifted_keywords, justification, keepsake
        drifted_keywords = payload.get("drifted_keywords") or []
        if not isinstance(drifted_keywords, list):
            drifted_keywords = []
        drifted_keywords = [str(kw).strip() for kw in drifted_keywords if str(kw).strip()]

        justification_en = str(payload.get("justification_en") or "").strip()
        justification_ar = str(payload.get("justification_ar") or "").strip()
        keepsake_en = str(payload.get("keepsake_en") or "").strip()
        keepsake_ar = str(payload.get("keepsake_ar") or "").strip()

        bundled["minds"][m.mind_key] = {
            "mind_id": m.mind_id,
            "prev_version": m.prev_version,
            "prev_drift_id": m.prev_drift_id,
            "replacements_en": reps_en,
            "replacements_ar": reps_ar,
            "summary_en": summary_en,
            "summary_ar": summary_ar,
            "drifted_keywords": drifted_keywords,
            "justification_en": justification_en,
            "justification_ar": justification_ar,
            "keepsake_en": keepsake_en,
            "keepsake_ar": keepsake_ar,
            "delta_json": json.dumps(en_delta_obj, ensure_ascii=False),
            "delta_json_ar": json.dumps(ar_patch_obj, ensure_ascii=False),
        }

    return bundled