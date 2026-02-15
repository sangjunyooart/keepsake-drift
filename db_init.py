# db_init.py
from __future__ import annotations

import argparse
import sqlite3


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1;",
        (name,),
    ).fetchone()
    return row is not None


def view_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='view' AND name=? LIMIT 1;",
        (name,),
    ).fetchone()
    return row is not None


def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return any(r["name"] == column for r in rows)


def add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    if column_exists(conn, table, column):
        return
    conn.execute(ddl)


def init_db(conn: sqlite3.Connection) -> None:
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ticks (
              tick_id     INTEGER PRIMARY KEY AUTOINCREMENT,
              started_at  TEXT NOT NULL DEFAULT (datetime('now')),
              status      TEXT NOT NULL DEFAULT 'running',
              notes       TEXT DEFAULT NULL
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS api_sources (
              source_id   INTEGER PRIMARY KEY AUTOINCREMENT,
              source_key  TEXT NOT NULL UNIQUE,
              name        TEXT NOT NULL,
              url         TEXT NOT NULL,
              kind        TEXT NOT NULL DEFAULT 'rss',
              is_enabled  INTEGER NOT NULL DEFAULT 1,
              meta_json   TEXT DEFAULT NULL,
              created_at  TEXT NOT NULL DEFAULT (datetime('now')),
              updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_api_sources_enabled ON api_sources(is_enabled);")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_events (
              event_id      INTEGER PRIMARY KEY AUTOINCREMENT,
              tick_id       INTEGER NOT NULL,
              source_id     INTEGER NOT NULL,
              event_time    TEXT NOT NULL,
              title         TEXT DEFAULT '',
              summary       TEXT DEFAULT '',
              content       TEXT DEFAULT '',
              url           TEXT DEFAULT '',
              external_id   TEXT DEFAULT NULL,
              content_hash  TEXT NOT NULL,
              embedding_id  INTEGER DEFAULT NULL,
              meta_json     TEXT DEFAULT NULL,
              created_at    TEXT NOT NULL DEFAULT (datetime('now')),
              FOREIGN KEY(tick_id) REFERENCES ticks(tick_id) ON DELETE CASCADE,
              FOREIGN KEY(source_id) REFERENCES api_sources(source_id) ON DELETE CASCADE
            );
            """
        )
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_raw_events_hash ON raw_events(content_hash);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_events_tick ON raw_events(tick_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_events_time ON raw_events(event_time);")

        if not table_exists(conn, "events") and not view_exists(conn, "events"):
            conn.execute(
                """
                CREATE VIEW events AS
                SELECT
                  event_id, tick_id, source_id, event_time,
                  title, summary, content, url,
                  external_id, content_hash, embedding_id, meta_json, created_at
                FROM raw_events;
                """
            )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS minds (
              mind_id    INTEGER PRIMARY KEY AUTOINCREMENT,
              mind_key   TEXT NOT NULL UNIQUE,
              label      TEXT NOT NULL DEFAULT '',
              created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mind_state (
              mind_id              INTEGER PRIMARY KEY,
              current_confidence   REAL NOT NULL DEFAULT 0.5,
              similarity_threshold REAL NOT NULL DEFAULT 0.5,
              alpha_up             REAL NOT NULL DEFAULT 0.1,
              beta_down            REAL NOT NULL DEFAULT 0.2,
              recent_weight        REAL NOT NULL DEFAULT 0.3,
              recent_window_n      INTEGER NOT NULL DEFAULT 8,
              last_tick_id         INTEGER DEFAULT NULL,
              last_drift_id        INTEGER DEFAULT NULL,
              updated_at           TEXT NOT NULL DEFAULT (datetime('now')),
              FOREIGN KEY(mind_id) REFERENCES minds(mind_id) ON DELETE CASCADE
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
              embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
              kind         TEXT NOT NULL,
              model        TEXT NOT NULL,
              dims         INTEGER NOT NULL,
              vector_blob  BLOB NOT NULL,
              vector_hash  TEXT NOT NULL UNIQUE,
              created_at   TEXT NOT NULL DEFAULT (datetime('now'))
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mind_prototypes (
              mind_id       INTEGER NOT NULL,
              version       INTEGER NOT NULL DEFAULT 1,
              seed_text     TEXT NOT NULL,
              embedding_id  INTEGER NOT NULL,
              weight        REAL NOT NULL DEFAULT 0.7,
              created_at    TEXT NOT NULL DEFAULT (datetime('now')),
              PRIMARY KEY(mind_id, version),
              FOREIGN KEY(mind_id) REFERENCES minds(mind_id) ON DELETE CASCADE,
              FOREIGN KEY(embedding_id) REFERENCES embeddings(embedding_id) ON DELETE CASCADE
            );
            """
        )

        # ---------------------------
        # drift_memory (canonical ledger)
        # Must match ticklib/pipeline.py insert list
        # ---------------------------
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS drift_memory (
              drift_id            INTEGER PRIMARY KEY AUTOINCREMENT,
              mind_id             INTEGER NOT NULL,
              tick_id             INTEGER NOT NULL,
              parent_drift_id     INTEGER DEFAULT NULL,
              drift_text          TEXT NOT NULL,
              summary_text        TEXT DEFAULT NULL,
              best_similarity     REAL NOT NULL DEFAULT 0.0,
              avg_similarity      REAL NOT NULL DEFAULT 0.0,
              grounded_ratio      REAL NOT NULL DEFAULT 0.0,
              hallucinate_ratio   REAL NOT NULL DEFAULT 1.0,
              confidence_before   REAL NOT NULL DEFAULT 0.5,
              confidence_after    REAL NOT NULL DEFAULT 0.5,
              hallucination_level REAL NOT NULL DEFAULT 0.0,
              embedding_id        INTEGER DEFAULT NULL,
              params_json         TEXT DEFAULT NULL,
              prompt_hash         TEXT DEFAULT NULL,
              created_at          TEXT NOT NULL DEFAULT (datetime('now')),
              version             INTEGER NOT NULL DEFAULT 0,
              FOREIGN KEY(mind_id) REFERENCES minds(mind_id) ON DELETE CASCADE,
              FOREIGN KEY(tick_id) REFERENCES ticks(tick_id) ON DELETE CASCADE,
              FOREIGN KEY(parent_drift_id) REFERENCES drift_memory(drift_id) ON DELETE SET NULL,
              FOREIGN KEY(embedding_id) REFERENCES embeddings(embedding_id) ON DELETE SET NULL
            );
            """
        )

        # Columns required by pipeline.py but missing in your older DB
        add_column_if_missing(
            conn, "drift_memory", "confidence_state",
            "ALTER TABLE drift_memory ADD COLUMN confidence_state TEXT NOT NULL DEFAULT 'unknown';"
        )
        add_column_if_missing(
            conn, "drift_memory", "hallucination_state",
            "ALTER TABLE drift_memory ADD COLUMN hallucination_state TEXT NOT NULL DEFAULT 'unknown';"
        )
        add_column_if_missing(
            conn, "drift_memory", "instability_reason",
            "ALTER TABLE drift_memory ADD COLUMN instability_reason TEXT DEFAULT NULL;"
        )
        add_column_if_missing(
            conn, "drift_memory", "keepsake_text",
            "ALTER TABLE drift_memory ADD COLUMN keepsake_text TEXT DEFAULT NULL;"
        )

        # Arabic + patch + underline payloads
        add_column_if_missing(
            conn, "drift_memory", "drift_text_ar",
            "ALTER TABLE drift_memory ADD COLUMN drift_text_ar TEXT NOT NULL DEFAULT '';"
        )
        add_column_if_missing(
            conn, "drift_memory", "summary_text_ar",
            "ALTER TABLE drift_memory ADD COLUMN summary_text_ar TEXT NOT NULL DEFAULT '';"
        )
        add_column_if_missing(
            conn, "drift_memory", "delta_json",
            "ALTER TABLE drift_memory ADD COLUMN delta_json TEXT NOT NULL DEFAULT '';"
        )
        add_column_if_missing(
            conn, "drift_memory", "ar_patch_json",
            "ALTER TABLE drift_memory ADD COLUMN ar_patch_json TEXT NOT NULL DEFAULT '';"
        )
        add_column_if_missing(
            conn, "drift_memory", "summary_ar_patch_json",
            "ALTER TABLE drift_memory ADD COLUMN summary_ar_patch_json TEXT NOT NULL DEFAULT '';"
        )

        conn.execute("CREATE INDEX IF NOT EXISTS idx_drift_memory_mind_ver ON drift_memory(mind_id, version);")

        # translation cache (needed for app.py/state translation)
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
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS ux_translation_cache_lang_text
            ON translation_cache(src_lang, tgt_lang, src_text);
            """
        )

        # originals
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS originals (
              temporality     TEXT PRIMARY KEY,
              original_text   TEXT NOT NULL,
              invariants_json TEXT DEFAULT NULL,
              meta_json       TEXT DEFAULT NULL,
              created_at      TEXT NOT NULL DEFAULT (datetime('now')),
              updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
            );
            """
        )
        add_column_if_missing(conn, "originals", "en", "ALTER TABLE originals ADD COLUMN en TEXT NOT NULL DEFAULT '';")
        add_column_if_missing(conn, "originals", "ar", "ALTER TABLE originals ADD COLUMN ar TEXT NOT NULL DEFAULT '';")
        conn.execute("UPDATE originals SET en = COALESCE(NULLIF(en,''), original_text) WHERE COALESCE(NULLIF(en,''), '') = '';")

        _seed_minds(conn)


def _seed_minds(conn: sqlite3.Connection) -> None:
    default_minds = [
        ("human", "Human Time"),
        ("liminal", "Liminal Time"),
        ("environment", "Environmental Time"),
        ("digital", "Digital Time"),
        ("infrastructure", "Infrastructure Time"),
        ("more_than_human", "More-than-human Time"),
    ]
    for key, label in default_minds:
        conn.execute("INSERT OR IGNORE INTO minds(mind_key, label) VALUES(?, ?);", (key, label))
    rows = conn.execute("SELECT mind_id FROM minds;").fetchall()
    for r in rows:
        conn.execute("INSERT OR IGNORE INTO mind_state(mind_id) VALUES(?);", (int(r["mind_id"]),))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True, help="Path to sqlite db, e.g. ./data/keepsake.sqlite")
    args = p.parse_args()

    conn = connect(args.db)
    try:
        init_db(conn)
        conn.commit()
        print("OK: schema ensured")
    finally:
        conn.close()


if __name__ == "__main__":
    main()