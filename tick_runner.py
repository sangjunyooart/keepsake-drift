# tick_runner.py
from __future__ import annotations

import argparse
import json
import os

from ticklib.pipeline import run_tick, DEFAULT_EMBED_DIMS


def _resolve_event_ids(db_path: str, tick_id: int, event_limit: int) -> list:
    """
    Look up recent event_ids from raw_events.
    Strategy: first try events from the same tick_id (ingest tick),
    then fall back to the most recent events across all ticks.
    """
    import storage

    with storage.connect(db_path) as conn:
        # First: events tied to this specific tick (if ingest created them under this tick_id)
        ids = storage.get_recent_event_ids(conn, tick_id=tick_id, limit=event_limit)
        if ids:
            return ids
        # Fallback: most recent events regardless of tick
        return storage.get_recent_event_ids(conn, limit=event_limit)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="./data/keepsake.sqlite")
    p.add_argument("--tick", dest="tick_id", type=int, required=False)
    p.add_argument("--tick_id", dest="tick_id_alt", type=int, required=False)
    p.add_argument("--event_limit", type=int, default=60)
    p.add_argument("--event_ids", type=str, default=None,
                   help="Comma-separated event IDs (overrides --event_limit lookup)")
    p.add_argument("--allow_fallback_recent", action="store_true")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="OpenAI API timeout in seconds (default: 120)")
    args = p.parse_args()

    tick_id = args.tick_id if args.tick_id is not None else args.tick_id_alt
    if tick_id is None:
        raise SystemExit("Missing required argument: --tick (or --tick_id)")

    # Resolve event_ids: explicit > DB lookup > empty
    if args.event_ids:
        event_ids = [int(x.strip()) for x in args.event_ids.split(",") if x.strip()]
    else:
        event_ids = _resolve_event_ids(args.db, int(tick_id), int(args.event_limit))

    if event_ids:
        print(f"Using {len(event_ids)} event_ids: {event_ids[:5]}{'...' if len(event_ids) > 5 else ''}")

    out = run_tick(
        db_path=args.db,
        tick_id=int(tick_id),
        event_ids=event_ids,
        embed_model=os.getenv("EMBED_MODEL", "local-embed-v0"),
        embed_dims=int(os.getenv("EMBED_DIMS", str(DEFAULT_EMBED_DIMS))),
        allow_fallback_recent=bool(args.allow_fallback_recent),
        timeout_seconds=float(args.timeout),
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
