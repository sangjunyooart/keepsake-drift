# bootstrap.py
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
import storage


def _load_original_memories(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _ensure_bootstrap_tick_zero(conn) -> None:
    """
    We want Drift 0 to be unambiguously bootstrap-seeded.
    So we explicitly create tick_id=0 (manual insert) and never use it for RSS ingestion.
    """
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(ticks);").fetchall()}
    if not cols:
        return

    if "started_at" in cols and "status" in cols and "notes" in cols:
        conn.execute(
            """
            INSERT OR IGNORE INTO ticks(tick_id, started_at, status, notes)
            VALUES (0, datetime('now'), 'bootstrap', 'seed Drift 0 from original_memories.json');
            """
        )
    elif "status" in cols:
        conn.execute("INSERT OR IGNORE INTO ticks(tick_id, status) VALUES (0, 'bootstrap');")
    else:
        conn.execute("INSERT OR IGNORE INTO ticks(tick_id) VALUES (0);")


def _mind_id_for_key(conn, mind_key: str) -> int:
    row = conn.execute("SELECT mind_id FROM minds WHERE mind_key=? LIMIT 1;", (mind_key,)).fetchone()
    if not row:
        raise RuntimeError(f"Missing mind in minds table: {mind_key}")
    return int(row["mind_id"])


def _upsert_original(conn, temporality: str, en: str, ar: str) -> None:
    en = (en or "").strip()
    ar = (ar or "").strip()
    temporality = (temporality or "").strip()
    if not temporality or not en:
        return

    # originals table has (temporality PK, original_text, en, ar, ...)
    conn.execute(
        """
        INSERT INTO originals(temporality, original_text, en, ar, updated_at)
        VALUES(?, ?, ?, ?, datetime('now'))
        ON CONFLICT(temporality) DO UPDATE SET
          original_text=excluded.original_text,
          en=excluded.en,
          ar=excluded.ar,
          updated_at=datetime('now');
        """,
        (temporality, en, en, ar),
    )


def _seed_drift0(
    conn, *, mind_id: int, tick_id: int, en: str, force: bool,
    invariables: Optional[List[Dict[str, str]]] = None,
) -> int:
    """
    Creates Drift 0 row for this mind:
    - version=0
    - tick_id=0
    - parent_drift_id=NULL
    - prompt_hash='bootstrap'
    - params_json includes a seed marker + invariables
    Returns drift_id.
    """
    en = (en or "").strip()
    if not en:
        raise RuntimeError("Cannot seed Drift 0 with empty text")

    if force:
        conn.execute("DELETE FROM drift_memory WHERE mind_id=? AND version=0;", (int(mind_id),))

    row = conn.execute(
        """
        SELECT drift_id
        FROM drift_memory
        WHERE mind_id=? AND version=0
        ORDER BY drift_id DESC
        LIMIT 1;
        """,
        (int(mind_id),),
    ).fetchone()
    if row:
        return int(row["drift_id"])

    params: Dict[str, Any] = {"seed": "original_memories.json", "kind": "axis", "version": 0}
    if invariables:
        params["invariables"] = invariables
        # Backward compat: legacy sensory_anchors as flat phrase list
        params["sensory_anchors"] = [inv["phrase"] for inv in invariables]
    conn.execute(
        """
        INSERT INTO drift_memory(
          mind_id, tick_id, parent_drift_id,
          drift_text, summary_text,
          best_similarity, avg_similarity, grounded_ratio, hallucinate_ratio,
          confidence_before, confidence_after, hallucination_level,
          embedding_id, params_json, prompt_hash, version,
          created_at
        ) VALUES (
          ?, ?, NULL,
          ?, NULL,
          0.0, 0.0, 1.0, 0.0,
          0.5, 0.5, 0.0,
          NULL, ?, 'bootstrap', 0,
          datetime('now')
        );
        """,
        (int(mind_id), int(tick_id), en, json.dumps(params, ensure_ascii=False)),
    )
    drift_id = int(conn.execute("SELECT last_insert_rowid();").fetchone()[0])
    return drift_id


def _update_mind_state(conn, *, mind_id: int, drift_id: int, tick_id: int) -> None:
    # mind_state has mind_id PK; ensure exists from db_init seed
    conn.execute(
        """
        UPDATE mind_state
        SET last_tick_id=?,
            last_drift_id=?,
            updated_at=datetime('now')
        WHERE mind_id=?;
        """,
        (int(tick_id), int(drift_id), int(mind_id)),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=config.SQLITE_PATH, help="sqlite path (default: config.SQLITE_PATH)")
    ap.add_argument("--force", action="store_true", help="overwrite existing Drift 0 rows")
    args = ap.parse_args()

    db_path = str(args.db)
    force = bool(args.force)

    # ensure schema + default minds
    storage.init_db(db_path)

    originals_path = Path(config.ORIGINAL_MEMORIES_PATH)
    originals = _load_original_memories(originals_path)

    with storage.connect(db_path) as conn:
        _ensure_bootstrap_tick_zero(conn)
        tick_id = 0

        seeded = []
        skipped = []

        # Collect all memory texts for invariable extraction
        memory_texts: Dict[str, str] = {}
        memory_ar: Dict[str, str] = {}
        for t in config.TEMPORALITIES:
            block = originals.get(t, {}) if isinstance(originals, dict) else {}
            en = (block.get("en") or "").strip()
            ar = (block.get("ar") or "").strip()
            if en:
                memory_texts[t] = en
                memory_ar[t] = ar

        # Extract invariables via OpenAI (1 bundled call for all minds)
        all_invariables: Dict[str, List[Dict[str, str]]] = {}
        if memory_texts:
            try:
                import drift_text_openai
                print("Extracting invariables via OpenAI...")
                all_invariables = drift_text_openai.extract_invariables_openai(memory_texts)
                inv_count = sum(len(v) for v in all_invariables.values())
                print(f"  Extracted {inv_count} invariables across {len(all_invariables)} minds")
            except Exception as e:
                print(f"  WARNING: Invariable extraction failed ({e}); proceeding without invariables")

        for t in config.TEMPORALITIES:
            en = memory_texts.get(t, "")
            ar = memory_ar.get(t, "")

            if not en:
                skipped.append(t)
                continue

            _upsert_original(conn, t, en, ar)

            # Store invariables in originals.invariants_json
            inv_list = all_invariables.get(t, [])
            if inv_list:
                inv_doc = {
                    "invariables": inv_list,
                    "extraction_model": "gpt-4.1-mini",
                    "extracted_at": datetime.now(timezone.utc).isoformat(),
                }
                conn.execute(
                    "UPDATE originals SET invariants_json = ? WHERE temporality = ?",
                    (json.dumps(inv_doc, ensure_ascii=False), t),
                )
                print(f"  {t}: {len(inv_list)} invariables stored")

            mind_id = _mind_id_for_key(conn, t)
            drift0_id = _seed_drift0(
                conn, mind_id=mind_id, tick_id=tick_id, en=en, force=force,
                invariables=inv_list,
            )
            _update_mind_state(conn, mind_id=mind_id, drift_id=drift0_id, tick_id=tick_id)

            seeded.append({"temporality": t, "mind_id": mind_id, "drift0_id": drift0_id})

    print("bootstrap complete")
    print(json.dumps({"db": db_path, "seeded": seeded, "skipped_missing_original": skipped}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())