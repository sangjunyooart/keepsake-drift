# ticklib/db_helpers.py
from __future__ import annotations

import sqlite3
from typing import Any, Optional


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    r = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=? LIMIT 1", (name,)
    ).fetchone()
    return r is not None


def _col_exists(conn: sqlite3.Connection, table: str, col: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row[1] == col for row in rows)


def _ensure_tick_row(conn: sqlite3.Connection, tick_id: int) -> None:
    if not _table_exists(conn, "ticks"):
        return
    try:
        conn.execute("INSERT OR IGNORE INTO ticks (tick_id) VALUES (?)", (int(tick_id),))
    except Exception:
        return


def _get_latest_row(conn: sqlite3.Connection, mind_id: int) -> Optional[sqlite3.Row]:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(drift_memory)").fetchall()}

    select_cols = ["drift_id", "mind_id", "tick_id", "parent_drift_id", "drift_text", "summary_text", "version", "created_at"]

    # Optional Arabic + patch fields if present
    for c in ("drift_text_ar", "summary_text_ar", "delta_json", "ar_patch_json", "summary_ar_patch_json"):
        if c in cols:
            select_cols.append(c)

    sql = f"""
    SELECT {", ".join(select_cols)}
    FROM drift_memory
    WHERE mind_id=?
    ORDER BY version DESC, drift_id DESC
    LIMIT 1
    """
    return conn.execute(sql, (int(mind_id),)).fetchone()


def _get_axis_row(conn: sqlite3.Connection, mind_id: int) -> Optional[sqlite3.Row]:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(drift_memory)").fetchall()}
    has_ar = "drift_text_ar" in cols

    # Try to find existing axis row (version 0)
    if has_ar:
        row = conn.execute(
            """
            SELECT drift_id, mind_id, tick_id, parent_drift_id, drift_text, drift_text_ar, version
            FROM drift_memory
            WHERE mind_id=? AND version=0
            ORDER BY drift_id ASC
            LIMIT 1
            """,
            (int(mind_id),),
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT drift_id, mind_id, tick_id, parent_drift_id, drift_text, version
            FROM drift_memory
            WHERE mind_id=? AND version=0
            ORDER BY drift_id ASC
            LIMIT 1
            """,
            (int(mind_id),),
        ).fetchone()

    if row is not None:
        return row

    # Clone earliest drift into version 0 if missing
    seed = conn.execute(
        """
        SELECT drift_id, drift_text, version
        FROM drift_memory
        WHERE mind_id=?
        ORDER BY version ASC, drift_id ASC
        LIMIT 1
        """,
        (int(mind_id),),
    ).fetchone()
    if seed is None:
        return None

    drift_text = seed["drift_text"] or ""

    conn.execute(
        """
        INSERT INTO drift_memory (mind_id, tick_id, parent_drift_id, drift_text, version)
        VALUES (?, ?, NULL, ?, 0)
        """,
        (int(mind_id), 0, str(drift_text)),
    )

    # Return the newly inserted axis row
    return conn.execute(
        """
        SELECT drift_id, mind_id, tick_id, parent_drift_id, drift_text, version
        FROM drift_memory
        WHERE mind_id=? AND version=0
        ORDER BY drift_id DESC
        LIMIT 1
        """,
        (int(mind_id),),
    ).fetchone()


def _store_delta_json(conn: sqlite3.Connection, drift_id: int, delta_json: str) -> None:
    if not _col_exists(conn, "drift_memory", "delta_json"):
        return
    conn.execute(
        "UPDATE drift_memory SET delta_json=? WHERE drift_id=?",
        (delta_json or "", int(drift_id)),
    )


def _store_arabic_fields(
    conn: sqlite3.Connection,
    drift_id: int,
    *,
    drift_text_ar: Optional[str],
    ar_patch_json: Optional[str],
    summary_text_ar: Optional[str],
    summary_ar_patch_json: Optional[str],
) -> None:
    cols = {
        c
        for c in ("drift_text_ar", "ar_patch_json", "summary_text_ar", "summary_ar_patch_json")
        if _col_exists(conn, "drift_memory", c)
    }
    if not cols:
        return

    sets = []
    vals: list[Any] = []
    if "drift_text_ar" in cols:
        sets.append("drift_text_ar=?")
        vals.append(drift_text_ar or "")
    if "ar_patch_json" in cols:
        sets.append("ar_patch_json=?")
        vals.append(ar_patch_json or "")
    if "summary_text_ar" in cols:
        sets.append("summary_text_ar=?")
        vals.append(summary_text_ar or "")
    if "summary_ar_patch_json" in cols:
        sets.append("summary_ar_patch_json=?")
        vals.append(summary_ar_patch_json or "")

    vals.append(int(drift_id))
    conn.execute(f"UPDATE drift_memory SET {', '.join(sets)} WHERE drift_id=?", vals)


def _store_keepsake_text(conn: sqlite3.Connection, drift_id: int, keepsake_text: Optional[str]) -> None:
    if not _col_exists(conn, "drift_memory", "keepsake_text"):
        return
    if keepsake_text is None:
        return
    conn.execute("UPDATE drift_memory SET keepsake_text=? WHERE drift_id=?", (keepsake_text, int(drift_id)))