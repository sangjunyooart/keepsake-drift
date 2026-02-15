# db_migrate_add_drifts.py
from __future__ import annotations

import argparse
import sqlite3


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    cols = set()
    for r in rows:
        # r: (cid, name, type, notnull, dflt_value, pk)
        cols.add(str(r[1]))
    return cols


def add_column_if_missing(conn: sqlite3.Connection, table: str, col: str, coltype: str, default_sql: str | None = None) -> bool:
    cols = table_columns(conn, table)
    if col in cols:
        return False
    if default_sql is None:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype};")
    else:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype} DEFAULT {default_sql};")
    return True


def ensure_index(conn: sqlite3.Connection, sql: str, name: str) -> None:
    conn.execute(sql)
    print(f"[migrate] index ok: {name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    args = ap.parse_args()

    conn = connect(args.db)
    try:
        with conn:
            # drift_memory additions (already in your migration history, safe to re-run)
            if add_column_if_missing(conn, "drift_memory", "keepsake_text", "TEXT"):
                print("[migrate] added drift_memory.keepsake_text")

            # NEW: delta_json for drift_memory
            if add_column_if_missing(conn, "drift_memory", "delta_json", "TEXT"):
                print("[migrate] added drift_memory.delta_json")

            ensure_index(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_drift_memory_mind_version ON drift_memory(mind_id, version);",
                "idx_drift_memory_mind_version",
            )
            ensure_index(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_drift_memory_tick_id ON drift_memory(tick_id);",
                "idx_drift_memory_tick_id",
            )

            # drifts additions (already in your migration history, safe to re-run)
            if add_column_if_missing(conn, "drifts", "keepsake_en", "TEXT"):
                print("[migrate] added drifts.keepsake_en")
            if add_column_if_missing(conn, "drifts", "keepsake_ar", "TEXT"):
                print("[migrate] added drifts.keepsake_ar")

            # NEW: delta_json for drifts
            if add_column_if_missing(conn, "drifts", "delta_json", "TEXT"):
                print("[migrate] added drifts.delta_json")

            ensure_index(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_drifts_temporality_version ON drifts(temporality, version);",
                "idx_drifts_temporality_version",
            )

        print(f"[migrate] ok: {args.db}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())