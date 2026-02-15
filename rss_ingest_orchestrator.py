#!/usr/bin/env python3
# rss_ingest_orchestrator.py
# Keepsake Drift — RSS ingest orchestrator
#
# Inserts into raw_events (with tick_id + source_id FKs).
# Ensures RSS sources are registered in api_sources.
# Creates a tick row for each ingest run.
#
# Usage:
#   python rss_ingest_orchestrator.py
#   python rss_ingest_orchestrator.py --db ./data/keepsake.sqlite --mode random --sources_per_run 2 --event_limit 24
#   python rss_ingest_orchestrator.py --mode cycle --sources_per_run 1 --event_limit 12

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import certifi
import feedparser
import requests

from config import SQLITE_PATH

CYCLE_STATE_PATH = "./data/rss_cycle_state.json"


# ----------------------------
# Models
# ----------------------------

@dataclass(frozen=True)
class OrchestratorSource:
    source_key: str
    name: str
    feed_url: str
    kind: str = "rss"
    is_enabled: int = 1
    meta: Optional[dict] = None


@dataclass(frozen=True)
class EventItem:
    source_key: str
    source_name: str
    published_at: str  # ISO8601 UTC
    title: str
    url: str
    summary: str
    content: str
    lang: str
    content_hash: str
    meta_json: str


# ----------------------------
# Utilities
# ----------------------------

def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_text(s: str) -> str:
    return " ".join((s or "").replace("\n", " ").replace("\r", " ").split()).strip()


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def to_iso_utc(dt_struct: Any) -> str:
    """
    feedparser gives time.struct_time for published_parsed/updated_parsed.
    Treat as UTC for deterministic storage.
    """
    try:
        if dt_struct:
            dt = datetime(*dt_struct[:6], tzinfo=timezone.utc)
            return dt.replace(microsecond=0).isoformat()
    except Exception:
        pass
    return utc_now_iso()


def load_cycle_state(path: str) -> dict:
    if not os.path.exists(path):
        return {"next_index": 0}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"next_index": 0}


def save_cycle_state(path: str, state: dict) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ----------------------------
# DB helpers
# ----------------------------

def connect(db_path: str) -> sqlite3.Connection:
    ensure_dir(os.path.dirname(db_path) or ".")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


def _ensure_sources_registered(conn: sqlite3.Connection) -> Dict[str, int]:
    """
    Ensure all RSS sources from rss_sources.py are in api_sources.
    Returns {source_key: source_id} mapping.
    """
    import storage
    storage.ensure_rss_sources(conn)

    rows = conn.execute("SELECT source_id, source_key FROM api_sources;").fetchall()
    return {r["source_key"]: int(r["source_id"]) for r in rows}


def _create_ingest_tick(conn: sqlite3.Connection) -> int:
    """Create a new tick row for this ingest run, return tick_id."""
    conn.execute(
        "INSERT INTO ticks(started_at, status, notes) VALUES (datetime('now'), 'ingest', 'rss_ingest_orchestrator');",
    )
    row = conn.execute("SELECT last_insert_rowid() AS id;").fetchone()
    return int(row["id"])


def insert_event(
    conn: sqlite3.Connection,
    ev: EventItem,
    *,
    tick_id: int,
    source_id: int,
) -> Optional[int]:
    """
    Insert into raw_events table with proper FK columns.
    Dedupe by UNIQUE(content_hash).
    """
    try:
        conn.execute(
            """
            INSERT INTO raw_events
              (tick_id, source_id, event_time, title, summary, content, url,
               external_id, content_hash, meta_json)
            VALUES
              (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                int(tick_id),
                int(source_id),
                ev.published_at,
                ev.title,
                ev.summary,
                ev.content,
                ev.url,
                json.loads(ev.meta_json).get("external_id", "") if ev.meta_json else "",
                ev.content_hash,
                ev.meta_json,
            ),
        )
        row = conn.execute("SELECT last_insert_rowid() AS id;").fetchone()
        return int(row["id"])
    except sqlite3.IntegrityError:
        # Duplicate content_hash — already ingested
        return None


# ----------------------------
# RSS fetch / parse
# ----------------------------

def fetch_rss(feed_url: str, timeout: int = 20) -> bytes:
    headers = {
        "User-Agent": "KeepsakeDriftRSS/1.7 (+local)",
        "Accept": "application/rss+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.1",
    }
    resp = requests.get(feed_url, headers=headers, timeout=timeout, verify=certifi.where())
    resp.raise_for_status()
    return resp.content


def parse_rss(xml_bytes: bytes, src: OrchestratorSource) -> List[EventItem]:
    feed = feedparser.parse(xml_bytes)

    feed_title = normalize_text(getattr(feed.feed, "title", "") or "")
    out: List[EventItem] = []

    for e in feed.entries:
        title = normalize_text(getattr(e, "title", "") or "") or "(untitled)"
        url = normalize_text(getattr(e, "link", "") or "")

        # Prefer full content; fallback to summary/description
        content_parts: List[str] = []
        if hasattr(e, "content") and getattr(e, "content", None):
            try:
                for c in e.content:
                    if isinstance(c, dict):
                        v = normalize_text(c.get("value", ""))
                    else:
                        v = normalize_text(str(c))
                    if v:
                        content_parts.append(v)
            except Exception:
                pass

        summary = normalize_text(getattr(e, "summary", "") or getattr(e, "description", "") or "")
        content = "\n\n".join([p for p in content_parts if p]) or summary

        if hasattr(e, "published_parsed") and getattr(e, "published_parsed", None):
            published_at = to_iso_utc(e.published_parsed)
        elif hasattr(e, "updated_parsed") and getattr(e, "updated_parsed", None):
            published_at = to_iso_utc(e.updated_parsed)
        else:
            published_at = utc_now_iso()

        external_id = getattr(e, "id", None)
        if external_id is not None:
            external_id = normalize_text(str(external_id)) or ""

        tags: List[str] = []
        if hasattr(e, "tags") and getattr(e, "tags", None):
            try:
                for t in e.tags:
                    if isinstance(t, dict) and t.get("term"):
                        tags.append(str(t["term"]))
                    elif hasattr(t, "term") and getattr(t, "term", None):
                        tags.append(str(t.term))
            except Exception:
                pass

        # Basic language hint (rss_sources may provide one; otherwise default en)
        lang = "en"
        if src.meta and isinstance(src.meta, dict):
            if (src.meta.get("lang") or "").strip():
                lang = str(src.meta["lang"]).strip()

        # Stable content hash for dedupe
        hash_basis = "|".join(
            [
                src.source_key,
                external_id or "",
                url,
                title,
                published_at,
                content[:5000],  # cap basis
            ]
        )
        content_hash = sha256_hex(hash_basis)

        meta = {
            "source_key": src.source_key,
            "source_name": src.name,
            "feed_title": feed_title,
            "external_id": external_id,
            "tags": tags,
            "raw": {
                "author": getattr(e, "author", None),
            },
        }

        meta_json = json.dumps(meta, ensure_ascii=False)

        out.append(
            EventItem(
                source_key=src.source_key,
                source_name=src.name,
                published_at=published_at,
                title=title,
                url=url or "",
                summary=summary or "",
                content=content or "",
                lang=lang,
                content_hash=content_hash,
                meta_json=meta_json,
            )
        )

    return out


# ----------------------------
# Source registry
# ----------------------------

def load_sources_from_registry() -> List[OrchestratorSource]:
    """
    Expects rss_sources.py defines RSS_SOURCES list with objects having:
      key, name, url, trust, lang, tags
    """
    try:
        import rss_sources  # type: ignore
    except Exception as ex:
        raise RuntimeError(f"cannot import rss_sources.py: {ex}") from ex

    sources_raw = getattr(rss_sources, "RSS_SOURCES", None)
    if not sources_raw:
        raise RuntimeError("rss_sources.RSS_SOURCES not found or empty")

    out: List[OrchestratorSource] = []
    for s in sources_raw:
        meta = {
            "trust": getattr(s, "trust", None),
            "lang": getattr(s, "lang", None),
            "tags": getattr(s, "tags", None),
        }
        out.append(
            OrchestratorSource(
                source_key=getattr(s, "key"),
                name=getattr(s, "name", getattr(s, "key")),
                feed_url=getattr(s, "url"),
                kind="rss",
                is_enabled=1,
                meta=meta,
            )
        )
    return out


def choose_sources(
    sources: List[OrchestratorSource],
    mode: str,
    k: int,
    seed: Optional[int],
    cycle_state_path: str = CYCLE_STATE_PATH,
) -> List[OrchestratorSource]:
    enabled = [s for s in sources if int(s.is_enabled) == 1]
    if not enabled:
        return []

    k = max(1, min(int(k), len(enabled)))

    if seed is not None:
        random.seed(seed)

    if mode == "random":
        return random.sample(enabled, k=k)

    state = load_cycle_state(cycle_state_path)
    idx = int(state.get("next_index", 0))

    chosen: List[OrchestratorSource] = []
    for i in range(k):
        chosen.append(enabled[(idx + i) % len(enabled)])

    state["next_index"] = (idx + k) % len(enabled)
    save_cycle_state(cycle_state_path, state)
    return chosen


# ----------------------------
# Orchestrator run
# ----------------------------

def run_ingest(
    *,
    db_path: str,
    mode: str,
    sources_per_run: int,
    event_limit: int,
    seed: Optional[int],
    timeout: int,
) -> dict:
    conn = connect(db_path)

    sources = load_sources_from_registry()
    chosen = choose_sources(sources, mode=mode, k=sources_per_run, seed=seed)

    fetched_total = 0
    inserted_total = 0
    inserted_ids: List[int] = []
    errors: List[str] = []
    tick_id: Optional[int] = None

    try:
        # Ensure all RSS sources are registered in api_sources
        source_map = _ensure_sources_registered(conn)
        conn.commit()

        # Create a tick for this ingest run
        tick_id = _create_ingest_tick(conn)
        conn.commit()

        for src in chosen:
            source_id = source_map.get(src.source_key)
            if source_id is None:
                errors.append(f"{src.source_key}: source_id not found in api_sources")
                continue

            try:
                xml = fetch_rss(src.feed_url, timeout=timeout)
                items = parse_rss(xml, src)
                fetched_total += len(items)

                for ev in items:
                    if inserted_total >= event_limit:
                        break
                    new_id = insert_event(conn, ev, tick_id=tick_id, source_id=source_id)
                    if new_id is not None:
                        inserted_total += 1
                        inserted_ids.append(new_id)

                if inserted_total >= event_limit:
                    break

            except Exception as ex:
                errors.append(f"{src.source_key}: {ex}")

        # Update tick status
        conn.execute(
            "UPDATE ticks SET status = 'ingested', notes = ? WHERE tick_id = ?;",
            (
                json.dumps({
                    "sources_used": [s.source_key for s in chosen],
                    "fetched": fetched_total,
                    "inserted": inserted_total,
                    "errors": errors,
                }, ensure_ascii=False),
                tick_id,
            ),
        )
        conn.commit()

    finally:
        conn.close()

    return {
        "db": db_path,
        "mode": mode,
        "tick_id": tick_id,
        "sources_used": [s.source_key for s in chosen],
        "fetched_total": fetched_total,
        "inserted_total": inserted_total,
        "inserted_event_ids": inserted_ids,
        "errors": errors,
        "ran_at": utc_now_iso(),
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Keepsake Drift RSS ingest orchestrator.")
    p.add_argument("--db", default=SQLITE_PATH, help="Path to SQLite DB")
    p.add_argument("--mode", choices=["random", "cycle"], default="random", help="Source selection mode")
    p.add_argument("--sources_per_run", type=int, default=6, help="How many RSS sources to ingest per run")
    p.add_argument("--event_limit", type=int, default=60, help="Max number of NEW events to insert per run")
    p.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    p.add_argument("--timeout", type=int, default=20, help="HTTP timeout seconds")
    args = p.parse_args(argv)

    result = run_ingest(
        db_path=args.db,
        mode=args.mode,
        sources_per_run=args.sources_per_run,
        event_limit=args.event_limit,
        seed=args.seed,
        timeout=args.timeout,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
