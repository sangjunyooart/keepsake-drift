# storage.py
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Set

import config
import db_init

# Expose this for older scripts that expect storage.SQLITE_PATH
SQLITE_PATH = config.SQLITE_PATH

# Separate database for translation cache to avoid locking conflicts
# The cache database is isolated from the main database so cache writes
# during ticks don't conflict with tick reads/writes
import os
from pathlib import Path
_cache_db_path = Path(config.SQLITE_PATH).parent / "translation_cache.sqlite"
CACHE_DB_PATH = str(_cache_db_path)


# ----------------------------
# Connection
# ----------------------------
@contextmanager
def connect(db_path: Optional[str] = None):
    path = db_path or config.SQLITE_PATH
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Optional[str] = None) -> None:
    with connect(db_path) as conn:
        db_init.init_db(conn)


# ----------------------------
# Introspection helpers
# ----------------------------
def _table_columns(conn: sqlite3.Connection, table: str) -> Set[str]:
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return {r["name"] for r in rows}


def _hash_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def _safe_has_table(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1;",
        (name,),
    ).fetchone()
    return row is not None


def _safe_ensure_translation_cache(conn: sqlite3.Connection) -> None:
    """
    Make translation_cache usable in both schema variants:

    Variant A (older):
      translation_cache(src_lang,tgt_lang,src_text,out_text) with UNIQUE(src_lang,tgt_lang,src_text)

    Variant B (newer):
      translation_cache(src_lang,tgt_lang,src_hash,src_text,out_text) with UNIQUE(src_lang,tgt_lang,src_hash)

    If table exists but UNIQUE constraint doesn't match what our insert uses,
    we fall back to INSERT without ON CONFLICT (duplicates are acceptable).
    """
    if not _safe_has_table(conn, "translation_cache"):
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS translation_cache (
              cache_id    INTEGER PRIMARY KEY AUTOINCREMENT,
              src_lang    TEXT NOT NULL,
              tgt_lang    TEXT NOT NULL,
              src_text    TEXT NOT NULL,
              out_text    TEXT NOT NULL,
              created_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );
            """
        )
        # Add a UNIQUE index for variant A
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS ux_translation_cache_lang_text
            ON translation_cache(src_lang, tgt_lang, src_text);
            """
        )
        return

    cols = _table_columns(conn, "translation_cache")

    # If table has src_hash, prefer Variant B unique index
    if "src_hash" in cols:
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS ux_translation_cache_lang_hash
            ON translation_cache(src_lang, tgt_lang, src_hash);
            """
        )
        return

    # Otherwise Variant A unique index
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_translation_cache_lang_text
        ON translation_cache(src_lang, tgt_lang, src_text);
        """
    )


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=8.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=8000;")
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def _connect_cache_db() -> sqlite3.Connection:
    """Connect to separate cache database to avoid main DB locking conflicts."""
    conn = sqlite3.connect(CACHE_DB_PATH, timeout=8.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=8000;")
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


# ----------------------------
# Minds / originals / drift_memory
# ----------------------------
def mind_id_for_temporality(conn: sqlite3.Connection, temporality: str) -> int:
    row = conn.execute(
        "SELECT mind_id FROM minds WHERE mind_key = ? LIMIT 1;",
        ((temporality or "").strip(),),
    ).fetchone()
    if not row:
        raise RuntimeError(f"mind not found for temporality: {temporality}")
    return int(row["mind_id"])


def get_original_axis(temporality: str, *, db_path: Optional[str] = None) -> Optional[Dict[str, str]]:
    temporality = (temporality or "").strip()
    if not temporality:
        return None
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT temporality, COALESCE(NULLIF(en,''), original_text) AS en, ar FROM originals WHERE temporality=?;",
            (temporality,),
        ).fetchone()
        if not row:
            return None
        return {"temporality": row["temporality"], "en": row["en"] or "", "ar": row["ar"] or ""}


def get_original(temporality: str, *, db_path: Optional[str] = None) -> Optional[Dict[str, str]]:
    return get_original_axis(temporality, db_path=db_path)


def get_latest_drift(temporality: str, *, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    temporality = (temporality or "").strip()
    if not temporality:
        return None

    with connect(db_path) as conn:
        if not _safe_has_table(conn, "drift_memory"):
            return None

        mind_id = mind_id_for_temporality(conn, temporality)
        cols = _table_columns(conn, "drift_memory")

        select_cols = [
            "drift_id",
            "mind_id",
            "tick_id",
            "parent_drift_id",
            "version",
            "drift_text",
            "summary_text",
            "created_at",
        ]

        optional_cols = [
            "drift_text_ar",
            "summary_text_ar",
            "ar_patch_json",
            "summary_ar_patch_json",
            "delta_json",
            "keepsake_text",
        ]
        for c in optional_cols:
            if c in cols:
                select_cols.append(c)

        row = conn.execute(
            f"""
            SELECT {", ".join(select_cols)}
            FROM drift_memory
            WHERE mind_id = ?
            ORDER BY version DESC, drift_id DESC
            LIMIT 1;
            """,
            (mind_id,),
        ).fetchone()

        return dict(row) if row else None


# ----------------------------
# Bootstrap helpers (needed by bootstrap.py)
# ----------------------------
def insert_bootstrap_tick(conn: sqlite3.Connection) -> int:
    cols = _table_columns(conn, "ticks")
    row = conn.execute("SELECT COALESCE(MAX(tick_id), -1) AS mx FROM ticks;").fetchone()
    next_id = int(row["mx"]) + 1

    if "status" in cols:
        conn.execute("INSERT INTO ticks(tick_id, status) VALUES (?, 'bootstrap');", (next_id,))
    else:
        conn.execute("INSERT INTO ticks(tick_id) VALUES (?);", (next_id,))
    return next_id


def seed_original_axis(temporality: str, en: str, ar: str, *, conn: sqlite3.Connection) -> None:
    temporality = (temporality or "").strip()
    en = (en or "").strip()
    ar = (ar or "").strip()
    if not temporality or not en:
        return

    cols = _table_columns(conn, "originals")
    if "en" in cols:
        conn.execute(
            """
            INSERT OR REPLACE INTO originals(temporality, original_text, en, ar, created_at)
            VALUES(?, ?, ?, ?, datetime('now'));
            """,
            (temporality, en, en, ar),
        )
    else:
        conn.execute(
            """
            INSERT OR REPLACE INTO originals(temporality, original_text, ar, created_at)
            VALUES(?, ?, ?, datetime('now'));
            """,
            (temporality, en, ar),
        )


def get_invariables(conn: sqlite3.Connection, temporality: str) -> list:
    """Retrieve pre-extracted invariables from originals.invariants_json for a temporality."""
    temporality = (temporality or "").strip()
    if not temporality:
        return []
    try:
        cols = _table_columns(conn, "originals")
        if "invariants_json" not in cols:
            return []
        row = conn.execute(
            "SELECT invariants_json FROM originals WHERE temporality = ?",
            (temporality,),
        ).fetchone()
        if not row or not row["invariants_json"]:
            return []
        import json
        data = json.loads(row["invariants_json"])
        return data.get("invariables", [])
    except Exception:
        return []


def drift_exists(conn: sqlite3.Connection, mind_id: int) -> bool:
    row = conn.execute(
        "SELECT 1 FROM drift_memory WHERE mind_id=? LIMIT 1;",
        (int(mind_id),),
    ).fetchone()
    return row is not None


# ----------------------------
# Translation cache (schema-adaptive)
# ----------------------------
def get_cached_translation(
    src_lang: str,
    tgt_lang: str,
    src_text: str,
    *,
    db_path: str,
    conn: Optional[sqlite3.Connection] = None,
) -> str:
    import logging
    logger = logging.getLogger(__name__)

    src_text = (src_text or "").strip()
    if not src_text:
        return ""

    own = None
    try:
        # Always use separate cache database (ignore db_path and conn params)
        # This prevents locking conflicts with main database during ticks
        own = _connect_cache_db()
        c = own

        _safe_ensure_translation_cache(c)
        cols = _table_columns(c, "translation_cache")

        if "src_hash" in cols:
            src_hash = _hash_text(src_text)
            row = c.execute(
                """
                SELECT out_text
                FROM translation_cache
                WHERE src_lang=? AND tgt_lang=? AND src_hash=?
                LIMIT 1
                """,
                (src_lang, tgt_lang, src_hash),
            ).fetchone()
        else:
            row = c.execute(
                """
                SELECT out_text
                FROM translation_cache
                WHERE src_lang=? AND tgt_lang=? AND src_text=?
                LIMIT 1
                """,
                (src_lang, tgt_lang, src_text),
            ).fetchone()

        result = (row["out_text"] if row and row["out_text"] else "") if row else ""

        # Log cache hit/miss for monitoring
        if result:
            logger.info(f"Translation cache HIT: {src_lang}->{tgt_lang} ({len(src_text)} chars)")
        else:
            logger.info(f"Translation cache MISS: {src_lang}->{tgt_lang} ({len(src_text)} chars)")

        return result
    finally:
        if own is not None:
            own.close()


def put_cached_translation(
    src_lang: str,
    tgt_lang: str,
    src_text: str,
    out_text: str,
    *,
    db_path: str,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    import logging
    logger = logging.getLogger(__name__)

    src_text = (src_text or "").strip()
    out_text = (out_text or "").strip()
    if not src_text or not out_text:
        return

    own = None
    try:
        # Always use separate cache database (ignore db_path and conn params)
        # This prevents locking conflicts with main database during ticks
        own = _connect_cache_db()
        c = own

        _safe_ensure_translation_cache(c)
        cols = _table_columns(c, "translation_cache")

        # Variant B: uses src_hash UNIQUE index
        if "src_hash" in cols:
            src_hash = _hash_text(src_text)
            try:
                c.execute(
                    """
                    INSERT INTO translation_cache (src_lang, tgt_lang, src_hash, src_text, out_text)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(src_lang, tgt_lang, src_hash) DO UPDATE SET
                      out_text=excluded.out_text,
                      src_text=excluded.src_text
                    """,
                    (src_lang, tgt_lang, src_hash, src_text, out_text),
                )
                logger.info(f"Translation cached: {src_lang}->{tgt_lang} ({len(src_text)} chars)")
            except sqlite3.OperationalError as e:
                logger.warning(f"Failed to cache translation (variant B): {e}")
                # If UNIQUE index is missing or mismatch, degrade gracefully
                try:
                    c.execute(
                        """
                        INSERT INTO translation_cache (src_lang, tgt_lang, src_hash, src_text, out_text)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (src_lang, tgt_lang, src_hash, src_text, out_text),
                    )
                    logger.info(f"Translation cached (fallback): {src_lang}->{tgt_lang}")
                except Exception as e2:
                    logger.error(f"Cache write failed completely: {e2}")

        # Variant A: uses src_text UNIQUE index
        else:
            try:
                c.execute(
                    """
                    INSERT INTO translation_cache (src_lang, tgt_lang, src_text, out_text)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(src_lang, tgt_lang, src_text) DO UPDATE SET
                      out_text=excluded.out_text
                    """,
                    (src_lang, tgt_lang, src_text, out_text),
                )
                logger.info(f"Translation cached: {src_lang}->{tgt_lang} ({len(src_text)} chars)")
            except sqlite3.OperationalError as e:
                logger.warning(f"Failed to cache translation (variant A): {e}")
                # If UNIQUE index is missing or mismatch, degrade gracefully
                try:
                    c.execute(
                        """
                        INSERT INTO translation_cache (src_lang, tgt_lang, src_text, out_text)
                        VALUES (?, ?, ?, ?)
                        """,
                        (src_lang, tgt_lang, src_text, out_text),
                    )
                    logger.info(f"Translation cached (fallback): {src_lang}->{tgt_lang}")
                except Exception as e2:
                    logger.error(f"Cache write failed completely: {e2}")

        if own is not None:
            own.commit()

    finally:
        if own is not None:
            own.close()


# ----------------------------
# RSS source + event helpers
# ----------------------------
def ensure_rss_sources(conn: sqlite3.Connection) -> int:
    """
    Register all RSS sources from rss_sources.py into api_sources table.
    Skips duplicates (source_key is UNIQUE). Returns count of newly inserted sources.
    """
    try:
        from rss_sources import RSS_SOURCES
    except ImportError:
        return 0

    inserted = 0
    for s in RSS_SOURCES:
        meta = json.dumps(
            {"trust": s.trust, "lang": s.lang, "tags": s.tags},
            ensure_ascii=False,
        )
        try:
            conn.execute(
                """
                INSERT INTO api_sources(source_key, name, url, kind, is_enabled, meta_json)
                VALUES (?, ?, ?, 'rss', 1, ?)
                ON CONFLICT(source_key) DO UPDATE SET
                  name=excluded.name, url=excluded.url, meta_json=excluded.meta_json,
                  updated_at=datetime('now');
                """,
                (s.key, s.name, s.url, meta),
            )
            inserted += 1
        except Exception:
            pass
    return inserted


def source_id_for_key(conn: sqlite3.Connection, source_key: str) -> Optional[int]:
    """Look up api_sources.source_id by source_key."""
    row = conn.execute(
        "SELECT source_id FROM api_sources WHERE source_key = ? LIMIT 1;",
        (source_key,),
    ).fetchone()
    return int(row["source_id"]) if row else None


def get_recent_event_ids(
    conn: sqlite3.Connection,
    *,
    tick_id: Optional[int] = None,
    limit: int = 24,
) -> List[int]:
    """
    Return recent event_ids from raw_events.
    If tick_id is given, return events for that tick; otherwise return the most recent events.
    """
    if tick_id is not None:
        rows = conn.execute(
            "SELECT event_id FROM raw_events WHERE tick_id = ? ORDER BY event_id DESC LIMIT ?;",
            (int(tick_id), int(limit)),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT event_id FROM raw_events ORDER BY event_id DESC LIMIT ?;",
            (int(limit),),
        ).fetchall()
    return [int(r["event_id"]) for r in rows]