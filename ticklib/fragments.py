# ticklib/fragments.py
from __future__ import annotations

from typing import Any, Dict, List


def _fallback_fragments_from_events(events: Dict[int, Dict[str, Any]], *, max_total: int = 10) -> List[str]:
    out: List[str] = []
    for _, e in events.items():
        for k in ("title", "content"):
            t = (e.get(k) or "").strip()
            if t:
                out.append(t[:260])
            if len(out) >= max_total:
                return out
    return out