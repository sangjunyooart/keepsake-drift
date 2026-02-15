#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import lens
import drift_text_openai  # OpenAI paragraph generator (with fallback to lens)


DEFAULT_DB_PATH = "./data/keepsake.sqlite"
TEMPORALITIES = ["human", "liminal", "environment", "digital", "infrastructure", "more_than_human"]
EMBED_DIM = 1024


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def connect(db_path: str) -> sqlite3.Connection:
    ensure_dir(os.path.dirname(db_path) or ".")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


def get_table_columns(conn: sqlite3.Connection, table: str) -> List[Dict[str, Any]]:
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    out = []
    for r in rows:
        out.append({"cid": r[0], "name": r[1], "type": r[2], "notnull": r[3], "dflt_value": r[4], "pk": r[5]})
    return out


def table_has_column(conn: sqlite3.Connection, table: str, col: str) -> bool:
    return col in {c["name"] for c in get_table_columns(conn, table)}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_text(s: str) -> str:
    return " ".join((s or "").replace("\n", " ").replace("\r", " ").split()).strip()


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def read_notes_json(notes: Optional[str]) -> Dict[str, Any]:
    if not notes:
        return {}
    try:
        return json.loads(notes)
    except Exception:
        return {}


def write_tick_notes(conn: sqlite3.Connection, tick_id: int, notes_obj: Dict[str, Any]) -> None:
    notes_json = json.dumps(notes_obj, ensure_ascii=False)
    with conn:
        conn.execute("UPDATE ticks SET notes = ? WHERE tick_id = ?;", (notes_json, tick_id))


# -------------------------
# Local embedding (simple)
# -------------------------

def _hash_token(tok: str) -> int:
    h = 2166136261
    for ch in tok:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def embed_text(text: str, dim: int = EMBED_DIM) -> List[float]:
    text = normalize_text(text).lower()
    if not text:
        return [0.0] * dim

    vec = [0.0] * dim
    tokens: List[str] = []
    buf: List[str] = []
    for ch in text:
        if ch.isalnum() or ch in ["_", "-"]:
            buf.append(ch)
        else:
            if buf:
                tokens.append("".join(buf))
                buf = []
    if buf:
        tokens.append("".join(buf))

    if not tokens:
        return [0.0] * dim

    for tok in tokens:
        idx = _hash_token(tok) % dim
        vec[idx] += 1.0

    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def cosine(a: List[float], b: List[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


# -------------------------
# Load tick + data
# -------------------------

def load_latest_ok_tick(conn: sqlite3.Connection) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT tick_id, started_at, status, notes FROM ticks WHERE status = 'ok' ORDER BY tick_id DESC LIMIT 1;"
    ).fetchone()


def load_tick_row(conn: sqlite3.Connection, tick_id: int) -> sqlite3.Row:
    row = conn.execute("SELECT tick_id, started_at, status, notes FROM ticks WHERE tick_id = ?;", (tick_id,)).fetchone()
    if not row:
        raise RuntimeError(f"tick_id {tick_id} not found.")
    return row


def load_tick_events(conn: sqlite3.Connection, tick_id: int, limit: int) -> List[sqlite3.Row]:
    q = """
    SELECT event_id, event_time, title, content, url, meta_json
    FROM raw_events
    WHERE tick_id = ?
    ORDER BY event_id DESC
    LIMIT ?;
    """
    return conn.execute(q, (tick_id, limit)).fetchall()


def load_originals(conn: sqlite3.Connection) -> Dict[str, str]:
    cols = {c["name"] for c in get_table_columns(conn, "originals")}
    if "temporality" not in cols or "original_text" not in cols:
        raise RuntimeError("originals table missing required columns (temporality, original_text).")

    rows = conn.execute("SELECT temporality, original_text FROM originals;").fetchall()
    out: Dict[str, str] = {str(r["temporality"]): str(r["original_text"] or "") for r in rows}
    for k in TEMPORALITIES:
        out.setdefault(k, "")
    return out


# -------------------------
# Minds helpers (mind_key)
# -------------------------

def ensure_minds_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS minds (
            mind_id INTEGER PRIMARY KEY AUTOINCREMENT,
            mind_key TEXT NOT NULL UNIQUE,
            label TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
    )


def get_or_create_mind_id(conn: sqlite3.Connection, mind_key: str) -> int:
    ensure_minds_table(conn)
    row = conn.execute("SELECT mind_id FROM minds WHERE mind_key = ? LIMIT 1;", (mind_key,)).fetchone()
    if row:
        return int(row["mind_id"])
    with conn:
        conn.execute("INSERT INTO minds(mind_key, label) VALUES (?, ?);", (mind_key, mind_key))
        return int(conn.execute("SELECT last_insert_rowid();").fetchone()[0])


def get_last_drift_id(conn: sqlite3.Connection, mind_id: int) -> Optional[int]:
    row = conn.execute(
        "SELECT drift_id FROM drifts WHERE mind_id = ? ORDER BY drift_id DESC LIMIT 1;",
        (mind_id,),
    ).fetchone()
    return int(row["drift_id"]) if row else None


def next_cycle_no(conn: sqlite3.Connection, mind_id: int) -> int:
    """
    Returns the next per-mind drift cycle number (1-based).
    - If drifts.cycle_no exists, uses MAX(cycle_no)+1.
    - Else uses COUNT(*)+1 (still per-mind).
    """
    if table_has_column(conn, "drifts", "cycle_no"):
        row = conn.execute(
            "SELECT COALESCE(MAX(cycle_no), 0) AS mx FROM drifts WHERE mind_id = ?;",
            (mind_id,),
        ).fetchone()
        return int(row["mx"] or 0) + 1

    row = conn.execute(
        "SELECT COUNT(*) AS n FROM drifts WHERE mind_id = ?;",
        (mind_id,),
    ).fetchone()
    return int(row["n"] or 0) + 1


# -------------------------
# Confidence / hallucination
# -------------------------

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def confidence_update(
    confidence_before: float,
    best_sim: float,
    avg_sim: float,
    isolation_flag: float,
    sim_floor: float = 0.10,
) -> Tuple[float, float]:
    """
    hallucination_level here is not "LLM hallucination", but an *isolation/confidence* signal:
    - similarity low AND isolation high -> very high hallucination_level
    - similarity low -> mid hallucination_level
    - isolation high -> mild instability even if similarity okay
    """
    c = float(confidence_before)
    weak_world_link = (best_sim < sim_floor)

    if weak_world_link and isolation_flag >= 1.0:
        c -= 0.10
    elif weak_world_link:
        c -= 0.06
    else:
        c += 0.04

    if avg_sim < 0.0:
        c -= 0.02

    c = clamp01(c)

    hallucination_level = 0.0
    if weak_world_link and isolation_flag >= 1.0:
        hallucination_level = 1.0
    elif weak_world_link:
        hallucination_level = 0.7
    else:
        hallucination_level = 0.2 if isolation_flag >= 1.0 else 0.0

    return c, hallucination_level


# -------------------------
# Evaluate tick
# -------------------------

def evaluate_tick(
    conn: sqlite3.Connection,
    tick_id: int,
    event_rows: List[sqlite3.Row],
    originals: Dict[str, str],
    top_k: int = 5,
    confidence_default: float = 0.5,
) -> Dict[str, Any]:
    events: List[Dict[str, Any]] = []
    for r in event_rows:
        title = str(r["title"] or "")
        content = str(r["content"] or "")
        url = str(r["url"] or "")
        event_text = normalize_text(title + "\n" + content)
        emb = embed_text(event_text)
        events.append({
            "event_id": int(r["event_id"]),
            "event_time": str(r["event_time"] or ""),
            "title": title,
            "url": url,
            "embedding": emb,
        })

    proto = {k: embed_text(originals.get(k, "")) for k in TEMPORALITIES}

    minds_tmp: Dict[str, Dict[str, Any]] = {}
    top_ids_by_mind: Dict[str, List[int]] = {}

    for mind in TEMPORALITIES:
        sims: List[Tuple[int, float]] = []
        for ev in events:
            s = cosine(proto[mind], ev["embedding"])
            sims.append((ev["event_id"], s))

        sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)
        top = sims_sorted[:top_k] if sims_sorted else []

        best_sim = float(top[0][1]) if top else 0.0
        avg_sim = float(sum(s for _, s in sims_sorted) / len(sims_sorted)) if sims_sorted else 0.0

        top_ids_by_mind[mind] = [eid for eid, _ in top]
        minds_tmp[mind] = {"best_similarity": best_sim, "avg_similarity": avg_sim, "top_events": top}

    # isolation: are my top events disjoint from other minds' top events?
    for mind in TEMPORALITIES:
        my_set = set(top_ids_by_mind[mind])
        others_union = set()
        for other_mind, ids in top_ids_by_mind.items():
            if other_mind == mind:
                continue
            others_union |= set(ids)
        isolation_flag = 1.0 if (len(my_set & others_union) == 0) else 0.0
        minds_tmp[mind]["isolation_flag"] = isolation_flag

    # previous confidence (from prior ticks notes, if present)
    prev = conn.execute(
        "SELECT notes FROM ticks WHERE tick_id < ? AND notes IS NOT NULL ORDER BY tick_id DESC LIMIT 1;",
        (tick_id,),
    ).fetchone()
    prev_notes = read_notes_json(prev["notes"]) if prev else {}
    prev_minds = prev_notes.get("minds", {}) if isinstance(prev_notes, dict) else {}

    minds_out: Dict[str, Any] = {}
    for mind in TEMPORALITIES:
        cb = float(confidence_default)
        if isinstance(prev_minds, dict) and mind in prev_minds and isinstance(prev_minds[mind], dict):
            cb = float(prev_minds[mind].get("confidence_after", confidence_default))

        best_sim = float(minds_tmp[mind]["best_similarity"])
        avg_sim = float(minds_tmp[mind]["avg_similarity"])
        isolation_flag = float(minds_tmp[mind]["isolation_flag"])

        ca, hallucination_level = confidence_update(cb, best_sim, avg_sim, isolation_flag)

        minds_out[mind] = {
            "confidence_before": round(cb, 4),
            "confidence_after": round(ca, 4),
            "best_similarity": round(best_sim, 6),
            "avg_similarity": round(avg_sim, 6),
            "isolation_flag": isolation_flag,
            "hallucination_level": round(hallucination_level, 4),
            "top_events": [[int(eid), float(sim)] for eid, sim in minds_tmp[mind]["top_events"]],
        }

    return {
        "tick_id": tick_id,
        "event_ids": [int(r["event_id"]) for r in event_rows],
        "evaluated_at": utc_now_iso(),
        "minds": minds_out,
        "events_index": {ev["event_id"]: {"title": ev["title"], "url": ev["url"]} for ev in events},
    }


# -------------------------
# Drift insert (schema-adaptive)
# -------------------------

def insert_drift_row(
    conn: sqlite3.Connection,
    *,
    mind_id: int,
    mind_key: str,
    tick_id: int,
    parent_drift_id: Optional[int],
    cycle_no: int,
    drift_text: str,
    summary_text: Optional[str],
    best_similarity: float,
    avg_similarity: float,
    grounded_ratio: float,
    hallucinate_ratio: float,
    confidence_before: float,
    confidence_after: float,
    hallucination_level: float,
    params_json: dict,
) -> int:
    cols = [c["name"] for c in get_table_columns(conn, "drifts")]
    cols_set = set(cols)

    # Always include cycle_no in params_json (even if column doesn't exist)
    params_json = dict(params_json or {})
    params_json["mind_key"] = mind_key
    params_json["cycle_no"] = int(cycle_no)

    row_data: Dict[str, Any] = {
        "mind_id": mind_id,
        "tick_id": tick_id,
        "parent_drift_id": parent_drift_id,
        "cycle_no": int(cycle_no),  # inserted only if column exists
        "drift_text": drift_text,
        "summary_text": summary_text,
        "best_similarity": float(best_similarity),
        "avg_similarity": float(avg_similarity),
        "grounded_ratio": float(grounded_ratio),
        "hallucinate_ratio": float(hallucinate_ratio),
        "confidence_before": float(confidence_before),
        "confidence_after": float(confidence_after),
        "hallucination_level": float(hallucination_level),
        "params_json": json.dumps(params_json, ensure_ascii=False),
        "prompt_hash": sha256_hex(drift_text),
        "embedding_id": None,
    }

    insert_cols = [k for k in row_data.keys() if k in cols_set]
    placeholders = ", ".join(["?"] * len(insert_cols))
    col_list = ", ".join(insert_cols)
    values = [row_data[k] for k in insert_cols]

    with conn:
        conn.execute(f"INSERT INTO drifts ({col_list}) VALUES ({placeholders});", tuple(values))
        row = conn.execute("SELECT last_insert_rowid() AS id;").fetchone()
        return int(row["id"])


def persist_drifts_for_tick(conn: sqlite3.Connection, eval_result: Dict[str, Any], originals: Dict[str, str]) -> Dict[str, int]:
    events_index = eval_result.get("events_index", {})
    minds = eval_result["minds"]
    drift_ids: Dict[str, int] = {}

    for mind_key, m in minds.items():
        mind_id = get_or_create_mind_id(conn, mind_key)
        parent = get_last_drift_id(conn, mind_id)
        cycle_no = next_cycle_no(conn, mind_id)

        top_event_ids = [eid for eid, _ in m["top_events"]]
        top_titles: List[str] = []
        for eid in top_event_ids:
            item = events_index.get(eid, {})
            t = item.get("title", "")
            if t:
                top_titles.append(str(t))

        hl = float(m["hallucination_level"])

        # OpenAI paragraph (fallback to rule-based lens)
        try:
            drift_text = drift_text_openai.generate_drift_paragraph_openai(
                mind_key=mind_key,
                original_text=originals.get(mind_key, ""),
                top_event_titles=top_titles,
                confidence_before=float(m["confidence_before"]),
                confidence_after=float(m["confidence_after"]),
                hallucination_level=hl,
            )
        except Exception:
            drift_text = lens.compose_drift_paragraph(
                mind_key=mind_key,
                original_text=originals.get(mind_key, ""),
                top_event_titles=top_titles,
                confidence_before=float(m["confidence_before"]),
                confidence_after=float(m["confidence_after"]),
                hallucination_level=hl,
            )

        summary_text = " | ".join([normalize_text(x) for x in top_titles[:3] if normalize_text(x)]) or None

        if hl >= 0.7:
            grounded_ratio = 0.0
            hallucinate_ratio = 1.0
        elif hl > 0.0:
            grounded_ratio = 0.5
            hallucinate_ratio = 0.5
        else:
            grounded_ratio = 1.0
            hallucinate_ratio = 0.0

        drift_id = insert_drift_row(
            conn,
            mind_id=mind_id,
            mind_key=mind_key,
            tick_id=int(eval_result["tick_id"]),
            parent_drift_id=parent,
            cycle_no=cycle_no,
            drift_text=drift_text,
            summary_text=summary_text,
            best_similarity=float(m["best_similarity"]),
            avg_similarity=float(m["avg_similarity"]),
            grounded_ratio=grounded_ratio,
            hallucinate_ratio=hallucinate_ratio,
            confidence_before=float(m["confidence_before"]),
            confidence_after=float(m["confidence_after"]),
            hallucination_level=hl,
            params_json={
                "mode": "event-driven",
                "top_events": m["top_events"],
                "isolation_flag": m["isolation_flag"],
                "event_ids": eval_result["event_ids"],
            },
        )
        drift_ids[mind_key] = drift_id

    return drift_ids


# -------------------------
# Main
# -------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate tick raw_events + write drifts (OpenAI paragraph with fallback).")
    ap.add_argument("--db", default=DEFAULT_DB_PATH)
    ap.add_argument("--tick_id", type=int, default=None, help="If omitted, uses latest ok tick")
    ap.add_argument("--event_limit", type=int, default=24)
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()

    conn = connect(args.db)
    try:
        if args.tick_id is None:
            tick_row = load_latest_ok_tick(conn)
            if not tick_row:
                raise RuntimeError("No ok tick found. Run RSS ingest first.")
        else:
            tick_row = load_tick_row(conn, int(args.tick_id))

        tick_id = int(tick_row["tick_id"])
        event_rows = load_tick_events(conn, tick_id, limit=args.event_limit)
        if not event_rows:
            raise RuntimeError(f"No raw_events for tick_id={tick_id}.")

        originals = load_originals(conn)

        eval_result = evaluate_tick(
            conn,
            tick_id=tick_id,
            event_rows=event_rows,
            originals=originals,
            top_k=args.top_k,
            confidence_default=0.5,
        )

        current_notes = read_notes_json(tick_row["notes"])
        if not isinstance(current_notes, dict):
            current_notes = {}

        current_notes["eval"] = {
            "mode": "event-driven",
            "event_limit": args.event_limit,
            "top_k": args.top_k,
            "evaluated_at": eval_result["evaluated_at"],
        }
        current_notes["event_ids"] = eval_result["event_ids"]
        current_notes["minds"] = eval_result["minds"]

        drift_ids = persist_drifts_for_tick(conn, eval_result, originals)
        current_notes["drift_ids"] = drift_ids

        write_tick_notes(conn, tick_id, current_notes)

        print(json.dumps({"status": "ok", "tick_id": tick_id, "drift_ids": drift_ids}, ensure_ascii=False, indent=2))
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())