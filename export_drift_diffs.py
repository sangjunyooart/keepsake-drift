#!/usr/bin/env python3
import json
import re
import sqlite3
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple, Optional

DB_PATH = Path("./data/keepsake.sqlite")
OUT_DIR = Path("./static/max_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PERSONAS = ["human", "liminal", "environment", "digital", "infrastructure", "more_than_human"]

WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)  # words + punctuation tokens

def tokenize_words(s: str) -> List[str]:
    s = (s or "").replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return WORD_RE.findall(s)

def diff_tokens(pre: List[str], post: List[str]) -> Dict:
    sm = SequenceMatcher(a=pre, b=post, autojunk=False)
    ops = []
    post_tags = ["keep"] * len(post)

    # Track deletions separately (ghost layer)
    deletions = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            ops.append({"op": "equal", "pre": [i1, i2], "post": [j1, j2]})
        elif tag == "insert":
            ops.append({"op": "insert", "post": [j1, j2]})
            for j in range(j1, j2):
                post_tags[j] = "add"
        elif tag == "delete":
            ops.append({"op": "delete", "pre": [i1, i2]})
            for i in range(i1, i2):
                deletions.append({"src_idx": i, "token": pre[i]})
        elif tag == "replace":
            ops.append({"op": "replace", "pre": [i1, i2], "post": [j1, j2]})
            for j in range(j1, j2):
                post_tags[j] = "change"
            for i in range(i1, i2):
                deletions.append({"src_idx": i, "token": pre[i]})

    # Build post token events (what Max will draw)
    post_events = []
    for idx, tok in enumerate(post):
        post_events.append({
            "idx": idx,
            "token": tok,
            "status": post_tags[idx],  # keep | add | change
        })

    return {
        "post_events": post_events,
        "deletions": deletions,
        "opcodes": ops,
    }

def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn

def latest_versions(conn: sqlite3.Connection, mind_key: str) -> Optional[Tuple[int, int]]:
    row = conn.execute(
        """
        SELECT MAX(d.version) AS v
        FROM drift_memory d
        JOIN minds m ON m.mind_id = d.mind_id
        WHERE m.mind_key = ?;
        """,
        (mind_key,),
    ).fetchone()
    if not row or row["v"] is None:
        return None
    v = int(row["v"])
    if v <= 0:
        return None
    return (v - 1, v)

def fetch_text_by_version(conn: sqlite3.Connection, mind_key: str, version: int) -> str:
    row = conn.execute(
        """
        SELECT d.drift_text
        FROM drift_memory d
        JOIN minds m ON m.mind_id = d.mind_id
        WHERE m.mind_key = ? AND d.version = ?
        ORDER BY d.drift_id DESC
        LIMIT 1;
        """,
        (mind_key, version),
    ).fetchone()
    return (row["drift_text"] if row and row["drift_text"] else "") or ""

def export_all(db_path: Path, out_dir: Path) -> None:
    with connect(db_path) as conn:
        for persona in PERSONAS:
            vv = latest_versions(conn, persona)
            if not vv:
                continue
            v_pre, v_post = vv
            pre_text = fetch_text_by_version(conn, persona, v_pre)
            post_text = fetch_text_by_version(conn, persona, v_post)

            pre_toks = tokenize_words(pre_text)
            post_toks = tokenize_words(post_text)
            diff = diff_tokens(pre_toks, post_toks)

            payload = {
                "persona": persona,
                "version_pre": v_pre,
                "version_post": v_post,
                "pre_text_len": len(pre_text),
                "post_text_len": len(post_text),
                "pre_token_count": len(pre_toks),
                "post_token_count": len(post_toks),
                "diff": diff,
            }

            (out_dir / f"{persona}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

if __name__ == "__main__":
    export_all(DB_PATH, OUT_DIR)
    print(f"Wrote diffs to {OUT_DIR.resolve()}")