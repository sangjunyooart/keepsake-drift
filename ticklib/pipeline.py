# ticklib/pipeline.py
from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_EMBED_DIMS = int(os.getenv("OPENAI_EMBED_DIMS", "1536"))

# Marker format for safe segment patching
SEG_OPEN = "<<<KDSEG:"
SEG_CLOSE = ">>>"
SEG_END = "<<<ENDSEG>>>"


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
    except Exception:
        pass
    try:
        conn.execute("PRAGMA busy_timeout=5000;")
    except Exception:
        pass
    return conn


def _table_cols(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _ensure_tick_row(conn: sqlite3.Connection, tick_id: int) -> None:
    cols = _table_cols(conn, "ticks")
    if "tick_id" not in cols:
        return
    if "created_at" in cols:
        conn.execute(
            "INSERT OR IGNORE INTO ticks (tick_id, created_at) VALUES (?, datetime('now'))",
            (int(tick_id),),
        )
    else:
        conn.execute("INSERT OR IGNORE INTO ticks (tick_id) VALUES (?)", (int(tick_id),))


def _fetch_minds(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    return conn.execute("SELECT mind_id, mind_key FROM minds ORDER BY mind_id ASC").fetchall()


def _fetch_latest_drift_row(conn: sqlite3.Connection, mind_id: int) -> Optional[sqlite3.Row]:
    cols = _table_cols(conn, "drift_memory")
    select_cols = ["drift_id", "version", "parent_drift_id", "drift_text", "summary_text", "tick_id", "created_at"]
    # Optional columns (many installs differ)
    for c in ("drift_text_ar", "summary_text_ar", "delta_json", "ar_patch_json", "params_json", "prompt_hash"):
        if c in cols:
            select_cols.append(c)

    sql = f"""
    SELECT {", ".join(select_cols)}
    FROM drift_memory
    WHERE mind_id = ?
    ORDER BY version DESC, drift_id DESC
    LIMIT 1
    """
    return conn.execute(sql, (int(mind_id),)).fetchone()


def _fetch_axis_row(conn: sqlite3.Connection, mind_id: int) -> Optional[sqlite3.Row]:
    # Axis policy: prefer version 0
    r0 = conn.execute(
        """
        SELECT drift_id, version, drift_text
        FROM drift_memory
        WHERE mind_id = ? AND version = 0
        ORDER BY drift_id ASC
        LIMIT 1
        """,
        (int(mind_id),),
    ).fetchone()
    if r0:
        return r0
    # Fallback: earliest row
    return conn.execute(
        """
        SELECT drift_id, version, drift_text
        FROM drift_memory
        WHERE mind_id = ?
        ORDER BY version ASC, drift_id ASC
        LIMIT 1
        """,
        (int(mind_id),),
    ).fetchone()


# -----------------------
# Segment selection (lens)
# -----------------------

SENSORY_WORDS = re.compile(
    r"\b(warm|cold|heat|humid|damp|dry|wind|breeze|smell|scent|taste|salt|dust|"
    r"light|dark|shadow|glow|noise|silence|echo|breath|skin|pulse|pressure|"
    r"stairs|footsteps|rain|fog|sun|night|dawn|dusk|afternoon|morning)\b",
    re.IGNORECASE,
)

PROPER_NOUNISH = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")

def _split_sentences(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    # Simple, stable splitter
    parts = re.split(r"(?<=[.!?])\s+", t)
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def _extract_drift_direction_str(entry: Any) -> str:
    """Extract drift_direction string from Stage 1 output (may be str or dict)."""
    if entry is None:
        return ""
    if isinstance(entry, dict):
        return str(entry.get("drift_direction") or "")
    return str(entry)


def _extract_infiltrating_imagery(entry: Any) -> List[str]:
    """Extract infiltrating_imagery list from Stage 1 output (concrete present-moment phrases)."""
    if entry is None:
        return []
    if isinstance(entry, dict):
        imagery = entry.get("infiltrating_imagery") or []
        if isinstance(imagery, list):
            return [str(x).strip() for x in imagery if str(x).strip()]
    return []


def _mind_decay_profile(mind_key: str) -> Dict[str, Any]:
    # decay rules are conceptual; used here for selection priorities
    mk = mind_key
    if mk == "human":
        return {"prefer": "proper_nouns", "preserve_sensory": True}
    if mk == "liminal":
        return {"prefer": "edges", "preserve_sensory": True}
    if mk in ("environment", "environmental"):
        return {"prefer": "cycles", "preserve_sensory": True}
    if mk == "digital":
        return {"prefer": "signals", "preserve_sensory": True}
    if mk == "infrastructure":
        return {"prefer": "routes", "preserve_sensory": True}
    if mk == "more_than_human":
        return {"prefer": "nonhuman", "preserve_sensory": True}
    return {"prefer": "proper_nouns", "preserve_sensory": True}


def select_segments_for_drift(
    *,
    mind_key: str,
    prev_text: str,
    tick_id: int,
    max_segments: int = 2,
    invariables: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """
    Returns a small list of segments eligible to blur/rewrite this tick.
    Key rule: we DO NOT let the model rewrite the whole drift.
    Uses invariable metadata when available, falls back to regex otherwise.
    """
    sents = _split_sentences(prev_text)
    if not sents:
        return []

    prof = _mind_decay_profile(mind_key)
    use_invariables = bool(invariables)

    scored: List[Tuple[float, int]] = []
    for i, s in enumerate(sents):
        score = 0.0
        s_lower = s.lower()

        if use_invariables:
            # Invariable-aware scoring
            has_proper = False
            has_sensory = False
            has_temporal = False
            for inv in invariables:  # type: ignore[union-attr]
                phrase = (inv.get("phrase") or "").lower()
                if not phrase or phrase not in s_lower:
                    continue
                cat = inv.get("category", "")
                if cat == "proper_noun":
                    has_proper = True
                elif cat == "sensory":
                    has_sensory = True
                elif cat == "temporal":
                    has_temporal = True

            # Sentences with proper nouns = good drift candidates (resolution loss)
            if has_proper:
                score += 2.0
            # Sentences with sensory invariables should be preserved
            if has_sensory:
                score -= 1.5
            # Temporal markers = moderate drift candidates
            if has_temporal:
                score += 0.5
        else:
            # Legacy regex fallback
            if PROPER_NOUNISH.search(s):
                score += 2.0
            if SENSORY_WORDS.search(s):
                score -= 1.5

        # Mind-specific nudges (lightweight, always active)
        p = prof.get("prefer")
        if p == "edges" and re.search(r"\b(threshold|edge|between|almost|waiting|hallway|door|border)\b", s, re.I):
            score += 1.0
        if p == "cycles" and re.search(r"\b(season|cycle|weather|rain|wind|soil|river|air)\b", s, re.I):
            score += 1.0
        if p == "signals" and re.search(r"\b(signal|packet|feed|latency|sync|buffer|index)\b", s, re.I):
            score += 1.0
        if p == "routes" and re.search(r"\b(route|corridor|grid|pipe|road|bridge|port|line)\b", s, re.I):
            score += 1.0
        if p == "nonhuman" and re.search(r"\b(species|moss|tree|bird|ocean|planet|evolution|migration)\b", s, re.I):
            score += 1.0

        scored.append((score, i))

    # Deterministic selection per tick/mind
    scored.sort(key=lambda x: (-x[0], x[1]))
    chosen_idx: List[int] = []

    # Choose best candidates
    for _, i in scored:
        if len(chosen_idx) >= max_segments:
            break
        chosen_idx.append(i)

    # Ensure we never select ALL sensorial lines; keep at least one sensorial anchor untouched if possible
    def _has_sensory(sent: str) -> bool:
        if use_invariables:
            sl = sent.lower()
            return any(
                (inv.get("phrase") or "").lower() in sl
                for inv in invariables  # type: ignore[union-attr]
                if inv.get("category") == "sensory"
            )
        return bool(SENSORY_WORDS.search(sent))

    if prof.get("preserve_sensory", True) and len(sents) > 2:
        if sum(1 for i in chosen_idx if _has_sensory(sents[i])) >= len(chosen_idx):
            for _, i in scored:
                if i in chosen_idx:
                    continue
                if not _has_sensory(sents[i]):
                    chosen_idx[-1] = i
                    break

    out: List[Dict[str, str]] = []
    for k, idx in enumerate(chosen_idx, start=1):
        out.append({"segment_id": f"{mind_key[:2]}{tick_id}_{k}", "text": sents[idx]})
    return out


def inject_segment_markers(prev_text: str, segments: List[Dict[str, str]]) -> str:
    """
    Replace exact sentence occurrences with markers wrapping the sentence text.
    This is robust and avoids needing start/end indices.
    """
    txt = prev_text or ""
    for seg in segments:
        sid = seg["segment_id"]
        s = seg["text"]
        if not s:
            continue
        marker_open = f"{SEG_OPEN}{sid}{SEG_CLOSE}"
        marker_block = f"{marker_open}{s}{SEG_END}"
        # Replace only first occurrence
        if s in txt:
            txt = txt.replace(s, marker_block, 1)
    return txt


def apply_replacements(marked_text: str, replacements: List[Dict[str, str]]) -> str:
    """
    Replace marker blocks by segment_id with generated text.
    Preserves original whitespace and punctuation as much as possible.
    """
    txt = marked_text or ""
    for rep in replacements:
        sid = (rep.get("segment_id") or "").strip()
        new_text = (rep.get("text") or "").strip()
        if not sid:
            continue

        # Replace content between markers for this sid (single block)
        pattern = re.escape(f"{SEG_OPEN}{sid}{SEG_CLOSE}") + r"(.*?)" + re.escape(SEG_END)
        txt = re.sub(pattern, new_text, txt, count=1, flags=re.DOTALL)

    # Remove any leftover markers (if a segment wasn't replaced)
    txt = re.sub(re.escape(SEG_OPEN) + r".*?" + re.escape(SEG_CLOSE), "", txt)
    txt = txt.replace(SEG_END, "")

    # Light cleanup only: collapse 3+ newlines, but do not flatten spaces everywhere
    txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
    return txt


def _extract_sensory_anchors(axis_en: str, max_lines: int = 2) -> List[str]:
    """
    LEGACY fallback: Extract 1–2 short anchor sentences via regex.
    Used only when no OpenAI-extracted invariables exist.
    """
    sents = _split_sentences(axis_en)
    if not sents:
        return []
    sens = [s for s in sents if SENSORY_WORDS.search(s)]
    picks = (sens[:max_lines] if sens else sents[:max_lines])
    return picks


def _load_invariables(conn: sqlite3.Connection, mind_key: str) -> List[Dict[str, str]]:
    """Load pre-extracted word/phrase-level invariables from originals.invariants_json."""
    try:
        import storage as _st
        return _st.get_invariables(conn, mind_key)
    except Exception:
        return []


def _invariables_to_anchors(invariables: List[Dict[str, str]]) -> List[str]:
    """Derive legacy sensory_anchors (flat phrase list) from invariables for backward compat."""
    return [inv["phrase"] for inv in invariables if inv.get("phrase")]


# -----------------------
# Insert row (schema-safe)
# -----------------------

def _insert_drift_memory_row(
    conn: sqlite3.Connection,
    *,
    mind_id: int,
    tick_id: int,
    parent_drift_id: Optional[int],
    version: int,
    drift_text: str,
    summary_text: str,
    drift_text_ar: str,
    summary_text_ar: str,
    delta_json: str,
    ar_patch_json: str,
    params_json: str,
    prompt_hash: str,
) -> int:
    cols = _table_cols(conn, "drift_memory")

    required = ["mind_id", "tick_id", "parent_drift_id", "drift_text", "summary_text", "version", "params_json", "prompt_hash"]
    values: List[Any] = [
        int(mind_id),
        int(tick_id),
        (int(parent_drift_id) if parent_drift_id is not None else None),
        str(drift_text or ""),
        (str(summary_text) if "summary_text" in cols else None),
        int(version),
        (str(params_json) if "params_json" in cols else None),
        (str(prompt_hash) if "prompt_hash" in cols else None),
    ]

    insert_cols: List[str] = []
    insert_vals: List[Any] = []
    for c, v in zip(required, values):
        if c in cols:
            insert_cols.append(c)
            insert_vals.append(v)

    optional_map = {
        "drift_text_ar": drift_text_ar,
        "summary_text_ar": summary_text_ar,
        "delta_json": delta_json,
        "ar_patch_json": ar_patch_json,
        # older installs might only have drift_text/summary_text; that's ok
    }
    for c, v in optional_map.items():
        if c in cols:
            insert_cols.append(c)
            insert_vals.append(str(v or ""))

    placeholders = ", ".join(["?"] * len(insert_cols))
    sql = f"INSERT INTO drift_memory ({', '.join(insert_cols)}) VALUES ({placeholders})"
    cur = conn.execute(sql, insert_vals)
    return int(cur.lastrowid)


# -----------------------
# Main entry
# -----------------------

@dataclass
class _MindRuntime:
    mind_id: int
    mind_key: str
    axis_en: str
    prev_en: str
    prev_ar: str
    prev_version: int
    prev_drift_id: Optional[int]
    selected_segments: List[Dict[str, str]]
    prev_marked_en: str
    sensory_anchors: List[str]
    invariables: List[Dict[str, str]]  # word/phrase-level invariables from OpenAI extraction


def run_tick(
    *,
    db_path: str,
    tick_id: int,
    allow_fallback_recent: bool = False,
    timeout_seconds: float = 120.0,
    text_model: Optional[str] = None,
    event_ids: Optional[List[int]] = None,
    **_ignored: Any,
) -> Dict[str, Any]:
    """
    v1.7 aligned tick:
    - Select a few eligible segments via lens
    - OpenAI returns replacements (EN+AR) for ONLY those segments (single call for all minds)
    - We reconstruct full drift text locally (axis continuity by construction)
    - We compute token deltas locally (existing JS highlighter contract)
    - We store EN (+ AR if schema supports)
    """
    event_ids = event_ids or []

    conn = _connect(db_path)
    try:
        _ensure_tick_row(conn, int(tick_id))
        minds_rows = _fetch_minds(conn)

        # Fetch event titles for justification prompt context
        event_titles: List[str] = []
        if event_ids:
            try:
                placeholders = ",".join("?" * len(event_ids))
                rows = conn.execute(
                    f"SELECT event_id, title FROM raw_events WHERE event_id IN ({placeholders})",
                    list(event_ids),
                ).fetchall()
                event_titles = [str(r["title"]) for r in rows if r["title"]]
            except Exception:
                pass

        runtimes: List[_MindRuntime] = []
        for mr in minds_rows:
            mind_id = int(mr["mind_id"])
            mind_key = str(mr["mind_key"])

            latest = _fetch_latest_drift_row(conn, mind_id)
            axis = _fetch_axis_row(conn, mind_id)

            axis_en = (axis["drift_text"] if axis and axis["drift_text"] is not None else "") if axis else ""
            prev_en = (latest["drift_text"] if latest and latest["drift_text"] is not None else "") if latest else axis_en
            prev_ar = ""
            if latest is not None and "drift_text_ar" in latest.keys():
                prev_ar = str(latest["drift_text_ar"] or "")
            prev_version = int(latest["version"]) if latest and latest["version"] is not None else 0
            prev_drift_id = int(latest["drift_id"]) if latest and latest["drift_id"] is not None else None

            # Load pre-extracted invariables; fall back to regex anchors if absent
            invariables = _load_invariables(conn, mind_key)
            if invariables:
                anchors = _invariables_to_anchors(invariables)
            else:
                anchors = _extract_sensory_anchors(axis_en, max_lines=2)

            selected = select_segments_for_drift(
                mind_key=mind_key,
                prev_text=prev_en,
                tick_id=int(tick_id),
                max_segments=int(os.getenv("KD_MAX_SEGMENTS_PER_MIND", "3")),
                invariables=invariables or None,
            )
            marked = inject_segment_markers(prev_en, selected)

            runtimes.append(
                _MindRuntime(
                    mind_id=mind_id,
                    mind_key=mind_key,
                    axis_en=axis_en,
                    prev_en=prev_en,
                    prev_ar=prev_ar,
                    prev_version=prev_version,
                    prev_drift_id=prev_drift_id,
                    selected_segments=selected,
                    prev_marked_en=marked,
                    sensory_anchors=anchors,
                    invariables=invariables,
                )
            )

        import drift_text_openai

        # Stage 1: Each temporality interprets headlines through its lens
        # Returns {mind_key: {drift_direction, infiltrating_imagery}} — pre-interpreted drift intent
        drift_directions: Dict[str, Any] = {}
        # Per-mind curated headlines (for Stage 2 as well)
        curated_headlines: Dict[str, List[str]] = {}
        mind_resonance: Dict[str, float] = {}  # 0.0-1.0 per mind
        if event_titles:
            try:
                import lens as _lens
                # Each mind selects headlines most resonant with its lens
                for r in runtimes:
                    hl_result = _lens.select_headlines_for_mind(
                        headlines=event_titles,
                        mind_key=r.mind_key,
                        max_headlines=6,
                        min_headlines=2,
                    )
                    curated_headlines[r.mind_key] = hl_result["headlines"]
                    mind_resonance[r.mind_key] = hl_result["resonance"]

                drift_directions = drift_text_openai.generate_lens_interpretations(
                    event_titles=event_titles,
                    minds=[
                        {
                            "mind_key": r.mind_key,
                            "perspective": _lens.PROTOTYPE_SEED_TEXT.get(r.mind_key, ""),
                            "decay_policy": _lens.decay_policy_for(r.mind_key),
                            "curated_headlines": curated_headlines.get(r.mind_key, event_titles[:6]),
                            "resonance": mind_resonance.get(r.mind_key, 0.5),
                        }
                        for r in runtimes
                    ],
                    model=text_model,
                    timeout_seconds=30.0,
                )
            except Exception as e:
                import logging
                logging.getLogger("pipeline").warning("Stage 1 lens interpretation failed: %s", e)

        # Stage 2: Generate drifts using lens interpretations as guidance
        # Each mind gets its own curated headlines (filtered by resonance)
        bundled = drift_text_openai.generate_bundled_drifts_openai(
            db_path=db_path,
            tick_id=int(tick_id),
            event_ids=list(event_ids),
            event_titles=event_titles,
            drift_directions=drift_directions,
            minds=[
                {
                    "mind_id": r.mind_id,
                    "mind_key": r.mind_key,
                    "axis_en": r.axis_en,
                    "prev_en": r.prev_en,
                    "prev_marked_en": r.prev_marked_en,
                    "sensory_anchors": r.sensory_anchors,
                    "invariables": r.invariables,
                    "selected_segments": r.selected_segments,
                    "prev_ar": r.prev_ar,
                    "prev_version": r.prev_version,
                    "prev_drift_id": r.prev_drift_id,
                    "event_titles": curated_headlines.get(r.mind_key, event_titles[:6]),
                    "resonance": mind_resonance.get(r.mind_key, 0.5),
                }
                for r in runtimes
            ],
            model=text_model,
            timeout_seconds=float(timeout_seconds),
        )

        # params_json is now built per-mind (in Phase 3) with trigger metadata
        _base_params = {
            "tick_id": int(tick_id),
            "text_model": (text_model or os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")),
            "allow_fallback_recent": bool(allow_fallback_recent),
            "max_segments_per_mind": int(os.getenv("KD_MAX_SEGMENTS_PER_MIND", "2")),
        }
        prompt_hash = str(bundled.get("prompt_hash") or "")

        # -------------------------------------------------------
        # Phase 1: Reconstruct final EN texts for all minds
        # -------------------------------------------------------
        rt_map = {r.mind_key: r for r in runtimes}

        # Intermediate store: {mind_key: {drift_en, drift_ar, recap_en, recap_ar, ...}}
        reconstructed: Dict[str, Dict[str, Any]] = {}

        for mind_key, payload in (bundled.get("minds") or {}).items():
            rt = rt_map.get(mind_key)
            if rt is None:
                continue

            prev_version = int(payload.get("prev_version") or rt.prev_version or 0)
            parent_drift_id = payload.get("prev_drift_id") or rt.prev_drift_id
            new_version = prev_version + 1 if prev_version >= 0 else 1

            # Model returns replacements; we reconstruct full text.
            reps_en = payload.get("replacements_en") or []
            reps_ar = payload.get("replacements_ar") or []

            drift_en = apply_replacements(rt.prev_marked_en, reps_en)
            # Arabic: do NOT reconstruct from reps_ar patched into English text
            # (that produces half-English, half-Arabic). Always let Phase 2
            # paired translation handle Arabic as a whole-text translation.
            drift_ar = ""
            recap_en = str(payload.get("summary_en") or "").strip()
            recap_ar = str(payload.get("summary_ar") or "").strip()

            # Invariant validation: log if sensory invariable concepts went missing
            # (prompt is primary enforcement; we don't append raw phrases since they're words, not sentences)
            if rt.invariables:
                import logging
                _log = logging.getLogger("pipeline")
                drift_lower = drift_en.lower()
                for inv in rt.invariables:
                    if inv.get("category") == "sensory":
                        phrase = (inv.get("phrase") or "").lower()
                        if phrase and phrase not in drift_lower:
                            _log.warning(
                                "Sensory invariable '%s' missing from drift for %s (tick %s)",
                                inv.get("phrase"), mind_key, tick_id,
                            )

            # New fields from expanded prompt contract
            drifted_keywords = payload.get("drifted_keywords") or []
            if not isinstance(drifted_keywords, list):
                drifted_keywords = []
            justification_en = str(payload.get("justification_en") or "").strip()
            justification_ar = str(payload.get("justification_ar") or "").strip()
            keepsake_en = str(payload.get("keepsake_en") or "").strip()
            keepsake_ar = str(payload.get("keepsake_ar") or "").strip()

            reconstructed[mind_key] = {
                "rt": rt,
                "drift_en": drift_en,
                "drift_ar": drift_ar,
                "recap_en": recap_en,
                "recap_ar": recap_ar,
                "prev_version": prev_version,
                "parent_drift_id": parent_drift_id,
                "new_version": new_version,
                "reps_en": reps_en,
                "drifted_keywords": drifted_keywords,
                "justification_en": justification_en,
                "justification_ar": justification_ar,
                "keepsake_en": keepsake_en,
                "keepsake_ar": keepsake_ar,
                "drift_direction": _extract_drift_direction_str(drift_directions.get(mind_key)),
                "infiltrating_imagery": _extract_infiltrating_imagery(drift_directions.get(mind_key)),
            }

        # -------------------------------------------------------
        # Phase 2: Paired translation for Arabic
        #   Translate (prev_en, drift_en) together so unchanged
        #   sentences get identical Arabic, and only drifted
        #   segments differ → accurate AR underlines.
        #   Also translate any missing recaps in a simple bundled call.
        # -------------------------------------------------------

        # 2a: Paired drift translation
        drift_pairs: Dict[str, Dict[str, str]] = {}
        for mind_key, rec in reconstructed.items():
            if rec["drift_en"] and not rec["drift_ar"]:
                drift_pairs[mind_key] = {
                    "prev_en": rec["rt"].prev_en or "",
                    "drift_en": rec["drift_en"],
                }

        # Import config at the beginning of Phase 2 (not inside conditionals)
        # to avoid scope issues when used in both Phase 2a and 2b
        import config as _cfg

        # DEBUG LOGGING
        import sys
        print(f"[DEBUG Phase 2a] drift_pairs count: {len(drift_pairs)}", file=sys.stderr)
        print(f"[DEBUG Phase 2a] drift_pairs keys: {list(drift_pairs.keys())}", file=sys.stderr)
        print(f"[DEBUG Phase 2] SECOND_LANG={_cfg.SECOND_LANG}, ENABLE_ARABIC={_cfg.ENABLE_ARABIC}", file=sys.stderr)

        if drift_pairs:
            try:

                pair_results = drift_text_openai.translate_drift_pairs_en_to_ar(
                    drift_pairs,
                    db_path=db_path,
                    target_lang=_cfg.SECOND_LANG,
                    timeout_seconds=180.0,  # 3 minutes for translating 6 drift pairs
                )

                print(f"[DEBUG Phase 2a] Translation returned {len(pair_results)} results", file=sys.stderr)

            except Exception as e:
                import traceback
                print(f"[ERROR Phase 2a] {type(e).__name__}: {str(e)}", file=sys.stderr)
                print(f"[ERROR Phase 2a] {traceback.format_exc()}", file=sys.stderr)
                pair_results = {}

            for mind_key, rec in reconstructed.items():
                pr = pair_results.get(mind_key)
                if pr:
                    if not rec["drift_ar"]:
                        rec["drift_ar"] = pr.get("drift_ar", "")
                    # Store paired prev_ar on the runtime so Phase 3 uses
                    # the translation-consistent prev_ar for delta computation
                    paired_prev_ar = pr.get("prev_ar", "")
                    if paired_prev_ar:
                        rec["rt"].prev_ar = paired_prev_ar
                        # Also write prev_ar back to the parent drift row
                        # so it's available for future ticks
                        if rec["rt"].prev_drift_id is not None:
                            try:
                                conn.execute(
                                    "UPDATE drift_memory SET drift_text_ar = ? WHERE drift_id = ? AND (drift_text_ar IS NULL OR length(trim(drift_text_ar)) = 0)",
                                    (paired_prev_ar, int(rec["rt"].prev_drift_id)),
                                )
                            except Exception:
                                pass

        # 2b: Bundled recap translation (recaps are short, no pairing needed)
        recap_texts: Dict[str, str] = {}
        for mind_key, rec in reconstructed.items():
            if rec["recap_en"] and not rec["recap_ar"]:
                recap_texts[f"{mind_key}__recap"] = rec["recap_en"]

        if recap_texts:
            try:
                recap_results = drift_text_openai.translate_bundled_en_to_ar(
                    recap_texts,
                    db_path=db_path,
                    target_lang=_cfg.SECOND_LANG,
                    timeout_seconds=120.0,  # 2 minutes for recaps (shorter than drift pairs)
                )
            except Exception:
                recap_results = {}

            for mind_key, rec in reconstructed.items():
                if not rec["recap_ar"]:
                    rec["recap_ar"] = recap_results.get(f"{mind_key}__recap", "")

        # 2c: Bundled keepsake translation (NEW - mirrors recap translation pattern)
        keepsake_texts: Dict[str, str] = {}
        for mind_key, rec in reconstructed.items():
            if rec["keepsake_en"] and not rec["keepsake_ar"]:
                keepsake_texts[f"{mind_key}__keepsake"] = rec["keepsake_en"]

        if keepsake_texts:
            try:
                keepsake_results = drift_text_openai.translate_bundled_en_to_ar(
                    keepsake_texts,
                    db_path=db_path,
                    target_lang=_cfg.SECOND_LANG,  # Respects pt-br, el, ar configuration
                    timeout_seconds=120.0,
                )
            except Exception:
                keepsake_results = {}

            for mind_key, rec in reconstructed.items():
                if not rec["keepsake_ar"]:
                    rec["keepsake_ar"] = keepsake_results.get(f"{mind_key}__keepsake", "")

        # -------------------------------------------------------
        # Phase 3: Compute authoritative deltas from final
        #   reconstructed text (not the approximation)
        # -------------------------------------------------------
        inserted: Dict[str, Any] = {}

        for mind_key, rec in reconstructed.items():
            rt = rec["rt"]
            drift_en = rec["drift_en"]
            drift_ar = rec["drift_ar"]

            # EN delta: prefer segment-aware (marks only replacement regions);
            # fall back to full-text SequenceMatcher diff if replacements
            # cannot be located in the reconstructed text.
            # Pass drifted_keywords for keyword-level underline narrowing.
            en_delta_obj = None
            if rec["reps_en"]:
                en_delta_obj = drift_text_openai.build_segment_aware_delta(
                    drift_en, rec["reps_en"], rt.selected_segments,
                    drifted_keywords=rec.get("drifted_keywords"),
                )
            if en_delta_obj is None:
                en_delta_obj = drift_text_openai.build_en_delta(rt.prev_en or "", drift_en)
            delta_json = json.dumps(en_delta_obj, ensure_ascii=False)

            # AR delta: derive from EN delta by proportional word-index mapping.
            # This ensures Arabic underlines match the same drifted segments
            # as English, regardless of translation variance.
            ar_patch_json = ""
            if drift_ar:
                ar_patch_obj = drift_text_openai.build_ar_patch_from_en_delta(
                    en_delta_obj, drift_en, drift_ar,
                )
                ar_patch_json = json.dumps(ar_patch_obj, ensure_ascii=False)

            # Build per-mind params_json with trigger metadata
            mind_params = dict(_base_params)
            mind_params["event_ids"] = list(event_ids)
            mind_params["selected_segments"] = [
                {"segment_id": seg.get("segment_id", ""), "text": seg.get("text", "")}
                for seg in (rt.selected_segments or [])
            ]
            mind_params["invariables"] = list(rt.invariables or [])
            mind_params["sensory_anchors"] = list(rt.sensory_anchors or [])  # backward compat
            mind_params["drifted_keywords"] = rec.get("drifted_keywords", [])
            mind_params["justification_en"] = rec.get("justification_en", "")
            mind_params["justification_ar"] = rec.get("justification_ar", "")
            mind_params["drift_direction"] = rec.get("drift_direction", "")
            mind_params["infiltrating_imagery"] = rec.get("infiltrating_imagery", [])
            mind_params["keepsake_text_ar"] = rec.get("keepsake_ar", "")
            mind_params["curated_headlines"] = curated_headlines.get(mind_key, [])
            mind_params["resonance"] = mind_resonance.get(mind_key, 0.5)
            params_json = json.dumps(mind_params, ensure_ascii=False)

            drift_id = _insert_drift_memory_row(
                conn,
                mind_id=rt.mind_id,
                tick_id=int(tick_id),
                parent_drift_id=(int(rec["parent_drift_id"]) if rec["parent_drift_id"] is not None else None),
                version=int(rec["new_version"]),
                drift_text=drift_en,
                summary_text=rec["recap_en"],
                drift_text_ar=drift_ar,
                summary_text_ar=rec["recap_ar"],
                delta_json=delta_json,
                ar_patch_json=ar_patch_json,
                params_json=params_json,
                prompt_hash=prompt_hash,
            )

            # Store keepsake_text (3rd-person narration) if available
            keepsake_text = rec.get("keepsake_en", "")
            if keepsake_text:
                try:
                    from ticklib.db_helpers import _store_keepsake_text
                    _store_keepsake_text(conn, drift_id, keepsake_text)
                except Exception:
                    pass

            inserted[mind_key] = {
                "drift_id": drift_id,
                "drift_version": int(rec["new_version"]),
                "segments_changed": len(rec["reps_en"]) if isinstance(rec["reps_en"], list) else 0,
                "has_arabic": bool(drift_ar),
            }

        # -------------------------------------------------------
        # Phase 3.5: Image generation (parallel, every N ticks)
        #   6 images sequentially ≈ 80s; parallel ≈ 15s
        # -------------------------------------------------------
        try:
            import image_gen
            import lens as _lens_img
            from concurrent.futures import ThreadPoolExecutor, as_completed

            # Collect jobs for minds that need images this tick
            img_jobs: List[Tuple[str, int, int, str, str, str, str, list, list]] = []
            for mind_key, rec in reconstructed.items():
                new_version = int(rec["new_version"])
                if not image_gen.should_generate(new_version):
                    continue
                drift_id_for_img = inserted.get(mind_key, {}).get("drift_id")
                # Include prev_drift_text so image can evolve from predecessor
                prev_drift_text = rec["rt"].prev_en if rec.get("rt") else ""
                # Extract sensory-category invariables as anchors for image grounding
                rt = rec["rt"]
                # Pass full invariable dicts (phrase + category) so image prompt
                # can give category-specific translation guidance (sensory → atmosphere,
                # proper_noun → landscape mood, temporal → light quality).
                # Capped at 8 total to prevent anchor dominance over present-moment scene.
                all_anchors = [
                    {"phrase": inv["phrase"], "category": inv.get("category", "sensory")}
                    for inv in (rt.invariables or [])
                    if inv.get("phrase")
                ][:8]
                # Present-moment imagery from Stage 1 lens interpretation
                infil_imagery = rec.get("infiltrating_imagery") or []
                img_jobs.append((
                    mind_key,
                    new_version,
                    drift_id_for_img,
                    rec["drift_en"],
                    rec.get("drift_direction", ""),
                    _lens_img.PROTOTYPE_SEED_TEXT.get(mind_key, ""),
                    prev_drift_text,
                    all_anchors,
                    infil_imagery,
                ))

            def _gen_one(job):
                mk, ver, did, dtxt, ddir, persp, prev_dtxt, anchors, infil = job
                path, img_prompt = image_gen.generate_drift_image(
                    mind_key=mk, version=ver,
                    drift_text=dtxt, drift_direction=ddir, perspective=persp,
                    prev_drift_text=prev_dtxt,
                    sensory_anchors=anchors,
                    infiltrating_imagery=infil,
                )
                return mk, ver, did, path, img_prompt

            # Run all image generations in parallel (max 6 threads)
            # img_results: {mind_key: (version, drift_id, path, image_prompt)}
            img_results: Dict[str, Tuple[int, Optional[int], Optional[str], str]] = {}
            if img_jobs:
                with ThreadPoolExecutor(max_workers=min(6, len(img_jobs))) as pool:
                    futures = {pool.submit(_gen_one, j): j[0] for j in img_jobs}
                    for fut in as_completed(futures):
                        try:
                            mk, ver, did, path, img_prompt = fut.result()
                            img_results[mk] = (ver, did, path, img_prompt)
                        except Exception as e:
                            import logging
                            logging.getLogger("pipeline").warning(
                                "Image gen thread failed for %s: %s", futures[fut], e,
                            )

            # Store image_path + image_prompt in params_json (DB writes are sequential, safe)
            for mind_key, (new_version, drift_id_for_img, img_path, img_prompt) in img_results.items():
                if drift_id_for_img and (img_path or img_prompt):
                    try:
                        row = conn.execute(
                            "SELECT params_json FROM drift_memory WHERE drift_id = ?",
                            (int(drift_id_for_img),),
                        ).fetchone()
                        if row:
                            pj = json.loads(row["params_json"] or "{}") if row["params_json"] else {}
                            if img_path:
                                pj["image_path"] = img_path
                            if img_prompt:
                                pj["image_prompt"] = img_prompt
                            conn.execute(
                                "UPDATE drift_memory SET params_json = ? WHERE drift_id = ?",
                                (json.dumps(pj, ensure_ascii=False), int(drift_id_for_img)),
                            )
                    except Exception as img_store_err:
                        import logging
                        logging.getLogger("pipeline").warning(
                            "Failed to store image_path for %s v%d: %s",
                            mind_key, new_version, img_store_err,
                        )

                if img_path:
                    inserted_info = inserted.get(mind_key)
                    if inserted_info:
                        inserted_info["image_path"] = img_path

        except Exception as img_gen_err:
            import logging
            logging.getLogger("pipeline").warning("Phase 3.5 image generation error: %s", img_gen_err)

        conn.commit()
        return {
            "tick_id": int(tick_id),
            "event_ids": list(event_ids),
            "minds": inserted,
            "status": "ok",
        }

    finally:
        try:
            conn.close()
        except Exception:
            pass