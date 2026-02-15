# db.py
from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List, Optional, Set


def connect_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
    except Exception:
        pass
    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> Set[str]:
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return {str(r["name"]) for r in rows}


def pick_event_ids_for_tick(conn: sqlite3.Connection, tick_id: int, limit: int) -> List[int]:
    rows = conn.execute(
        """
        SELECT event_id
        FROM raw_events
        WHERE tick_id = ?
        ORDER BY event_time DESC, event_id DESC
        LIMIT ?;
        """,
        (int(tick_id), int(limit)),
    ).fetchall()
    return [int(r["event_id"]) for r in rows]


def pick_recent_event_ids(conn: sqlite3.Connection, limit: int) -> List[int]:
    rows = conn.execute(
        """
        SELECT event_id
        FROM raw_events
        ORDER BY event_time DESC, event_id DESC
        LIMIT ?;
        """,
        (int(limit),),
    ).fetchall()
    return [int(r["event_id"]) for r in rows]


def load_event_texts(conn: sqlite3.Connection, event_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    if not event_ids:
        return {}
    placeholders = ",".join(["?"] * len(event_ids))
    rows = conn.execute(
        f"""
        SELECT event_id, event_time, title, content, url, source_id, embedding_id
        FROM raw_events
        WHERE event_id IN ({placeholders});
        """,
        event_ids,
    ).fetchall()

    out: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        out[int(r["event_id"])] = {
            "event_time": r["event_time"],
            "title": r["title"] or "",
            "content": r["content"] or "",
            "url": r["url"] or "",
            "source_id": r["source_id"],
            "embedding_id": r["embedding_id"],
        }
    return out


def load_minds(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT m.mind_id, m.mind_key,
               ms.current_confidence, ms.similarity_threshold, ms.alpha_up, ms.beta_down,
               ms.recent_weight, ms.recent_window_n
        FROM minds m
        JOIN mind_state ms ON ms.mind_id = m.mind_id
        ORDER BY m.mind_id ASC;
        """
    ).fetchall()

    minds: List[Dict[str, Any]] = []
    for r in rows:
        minds.append(
            dict(
                mind_id=int(r["mind_id"]),
                mind_key=str(r["mind_key"]),
                confidence=float(r["current_confidence"]),
                threshold=float(r["similarity_threshold"]),
                alpha_up=float(r["alpha_up"]),
                beta_down=float(r["beta_down"]),
                recent_weight=float(r["recent_weight"]),
                recent_n=int(r["recent_window_n"]),
            )
        )
    return minds


def get_latest_drift_row(conn: sqlite3.Connection, mind_id: int) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        """
        SELECT *
        FROM drift_memory
        WHERE mind_id = ?
        ORDER BY version DESC, drift_id DESC
        LIMIT 1;
        """,
        (int(mind_id),),
    ).fetchone()
    return dict(row) if row else None


def next_drift_version(conn: sqlite3.Connection, mind_id: int) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(version), -1) AS vmax FROM drift_memory WHERE mind_id=?;",
        (int(mind_id),),
    ).fetchone()
    return int(row["vmax"]) + 1


def upsert_embedding(
    conn: sqlite3.Connection,
    *,
    kind: str,
    model: str,
    dims: int,
    vector_blob: bytes,
    vector_hash: str,
) -> int:
    row = conn.execute(
        "SELECT embedding_id FROM embeddings WHERE vector_hash=? LIMIT 1;",
        (vector_hash,),
    ).fetchone()
    if row:
        return int(row["embedding_id"])

    conn.execute(
        """
        INSERT INTO embeddings(kind, model, dims, vector_blob, vector_hash)
        VALUES (?, ?, ?, ?, ?);
        """,
        (kind, model, int(dims), vector_blob, vector_hash),
    )
    return int(conn.execute("SELECT last_insert_rowid();").fetchone()[0])


def load_embedding_blob(conn: sqlite3.Connection, embedding_id: int) -> bytes:
    row = conn.execute(
        "SELECT vector_blob FROM embeddings WHERE embedding_id=? LIMIT 1;",
        (int(embedding_id),),
    ).fetchone()
    if not row:
        raise KeyError(f"embedding_id not found: {embedding_id}")
    return row["vector_blob"]


def set_raw_event_embedding(conn: sqlite3.Connection, event_id: int, embedding_id: int) -> None:
    conn.execute(
        "UPDATE raw_events SET embedding_id=? WHERE event_id=?;",
        (int(embedding_id), int(event_id)),
    )


def ensure_tick_row(conn: sqlite3.Connection, tick_id: int) -> None:
    cols = _table_columns(conn, "ticks")
    if "status" in cols:
        conn.execute(
            "INSERT OR IGNORE INTO ticks(tick_id, status) VALUES (?, 'running');",
            (int(tick_id),),
        )
    else:
        conn.execute("INSERT OR IGNORE INTO ticks(tick_id) VALUES (?);", (int(tick_id),))


def insert_drift_memory(
    conn: sqlite3.Connection,
    *,
    mind_id: int,
    tick_id: int,
    parent_drift_id: Optional[int],
    drift_text: str,
    summary_text: Optional[str],
    keepsake_text: Optional[str] = None,

    # --- metrics (NOW OPTIONAL / SAFE DEFAULTS) ---
    best_similarity: float = 0.0,
    avg_similarity: float = 0.0,
    grounded_ratio: float = 0.0,
    hallucinate_ratio: float = 1.0,
    confidence_before: float = 0.5,
    confidence_after: float = 0.5,
    hallucination_level: float = 0.0,
    embedding_id: Optional[int] = None,
    params: Optional[Dict[str, Any]] = None,
    prompt_hash: str = "",

    # --- extras (unchanged) ---
    drift_text_ar: Optional[str] = None,
    summary_text_ar: Optional[str] = None,
    keepsake_text_ar: Optional[str] = None,
    instability_reason: Optional[str] = None,
    invariants_phrases: Optional[List[str]] = None,
    invariant_phrases: Optional[List[str]] = None,
    **extra: Any,
) -> int:
    """
    Insert a new drift_memory row.

    This function is schema-adaptive:
    - If drift_memory.keepsake_text exists, we store it in the column.
    - Otherwise, we store it in params_json.
    """
    ensure_tick_row(conn, int(tick_id))

    cols = _table_columns(conn, "drift_memory")
    v = next_drift_version(conn, int(mind_id))

    params_obj: Dict[str, Any] = dict(params or {})

    if drift_text_ar is not None:
        params_obj["drift_text_ar"] = drift_text_ar
    if summary_text_ar is not None:
        params_obj["summary_text_ar"] = summary_text_ar
    if keepsake_text_ar is not None:
        params_obj["keepsake_text_ar"] = keepsake_text_ar
    if instability_reason is not None:
        params_obj["instability_reason"] = instability_reason

    inv = invariants_phrases if invariants_phrases is not None else invariant_phrases
    if inv is not None:
        params_obj["invariants_phrases"] = [str(x).strip() for x in inv if str(x).strip()]

    if keepsake_text is not None and "keepsake_text" not in cols:
        params_obj["keepsake_text"] = keepsake_text

    if extra:
        params_obj.setdefault("_extra", {})
        for k, val in extra.items():
            params_obj["_extra"][k] = val

    if "keepsake_text" in cols:
        conn.execute(
            """
            INSERT INTO drift_memory(
              mind_id, tick_id, parent_drift_id,
              drift_text, summary_text, keepsake_text,
              best_similarity, avg_similarity, grounded_ratio, hallucinate_ratio,
              confidence_before, confidence_after, hallucination_level,
              embedding_id, params_json, prompt_hash, version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                int(mind_id),
                int(tick_id),
                int(parent_drift_id) if parent_drift_id is not None else None,
                drift_text or "",
                summary_text,
                (keepsake_text or "").strip() if keepsake_text is not None else None,
                float(best_similarity),
                float(avg_similarity),
                float(grounded_ratio),
                float(hallucinate_ratio),
                float(confidence_before),
                float(confidence_after),
                float(hallucination_level),
                int(embedding_id) if embedding_id is not None else None,
                json.dumps(params_obj, ensure_ascii=False),
                prompt_hash or "",
                int(v),
            ),
        )
    else:
        conn.execute(
            """
            INSERT INTO drift_memory(
              mind_id, tick_id, parent_drift_id,
              drift_text, summary_text,
              best_similarity, avg_similarity, grounded_ratio, hallucinate_ratio,
              confidence_before, confidence_after, hallucination_level,
              embedding_id, params_json, prompt_hash, version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                int(mind_id),
                int(tick_id),
                int(parent_drift_id) if parent_drift_id is not None else None,
                drift_text or "",
                summary_text,
                float(best_similarity),
                float(avg_similarity),
                float(grounded_ratio),
                float(hallucinate_ratio),
                float(confidence_before),
                float(confidence_after),
                float(hallucination_level),
                int(embedding_id) if embedding_id is not None else None,
                json.dumps(params_obj, ensure_ascii=False),
                prompt_hash or "",
                int(v),
            ),
        )

    return int(conn.execute("SELECT last_insert_rowid();").fetchone()[0])


def upsert_drift_evidence(
    conn: sqlite3.Connection,
    *,
    drift_id: int,
    event_id: int,
    similarity: float,
    used_in_grounded: int,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO drift_evidence(drift_id, event_id, similarity, used_in_grounded, weight)
        VALUES (?, ?, ?, ?, 1.0);
        """,
        (int(drift_id), int(event_id), float(similarity), int(used_in_grounded)),
    )


def update_mind_state(
    conn: sqlite3.Connection,
    *,
    mind_id: int,
    tick_id: int,
    drift_id: int,
    confidence: float,
) -> None:
    conn.execute(
        """
        UPDATE mind_state
        SET current_confidence = ?, last_tick_id = ?, last_drift_id = ?, updated_at = datetime('now')
        WHERE mind_id = ?;
        """,
        (float(confidence), int(tick_id), int(drift_id), int(mind_id)),
    )


# ----------------------------
# Drift post-update helpers
# ----------------------------

def _get_params_json(conn: sqlite3.Connection, drift_id: int) -> Dict[str, Any]:
    row = conn.execute(
        "SELECT params_json FROM drift_memory WHERE drift_id=? LIMIT 1;",
        (int(drift_id),),
    ).fetchone()
    if not row:
        return {}
    raw = row["params_json"] or ""
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _set_params_json(conn: sqlite3.Connection, drift_id: int, obj: Dict[str, Any]) -> None:
    conn.execute(
        "UPDATE drift_memory SET params_json=? WHERE drift_id=?;",
        (json.dumps(obj or {}, ensure_ascii=False), int(drift_id)),
    )


def set_drift_embedding(conn: sqlite3.Connection, drift_id: int, embedding_id: Optional[int]) -> None:
    """
    Compatibility for ticklib/pipeline.py
    """
    conn.execute(
        "UPDATE drift_memory SET embedding_id=? WHERE drift_id=?;",
        (int(embedding_id) if embedding_id is not None else None, int(drift_id)),
    )


def set_drift_patch_json(conn: sqlite3.Connection, drift_id: int, patch_json: str) -> None:
    """
    Store Arabic patch JSON. If the column exists, use it. Otherwise store in params_json.
    """
    cols = _table_columns(conn, "drift_memory")
    if "ar_patch_json" in cols:
        conn.execute(
            "UPDATE drift_memory SET ar_patch_json=? WHERE drift_id=?;",
            (patch_json or "", int(drift_id)),
        )
        return
    obj = _get_params_json(conn, drift_id)
    obj["ar_patch_json"] = patch_json or ""
    _set_params_json(conn, drift_id, obj)


def set_drift_delta_json(conn: sqlite3.Connection, drift_id: int, delta_json: str) -> None:
    """
    Store English delta JSON. If the column exists, use it. Otherwise store in params_json.
    """
    cols = _table_columns(conn, "drift_memory")
    if "delta_json" in cols:
        conn.execute(
            "UPDATE drift_memory SET delta_json=? WHERE drift_id=?;",
            (delta_json or "", int(drift_id)),
        )
        return
    obj = _get_params_json(conn, drift_id)
    obj["delta_json"] = delta_json or ""
    _set_params_json(conn, drift_id, obj)