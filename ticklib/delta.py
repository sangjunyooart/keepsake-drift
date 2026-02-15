# ticklib/delta.py
from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Dict, List, Tuple


# =========================
# Tokenization (MUST match JS)
# =========================
# EN: roughly matches your JS RE_EN word group behavior
_RE_EN_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

# Unicode “word” tokens (matches JS /[^\W_]+/gu):
# - any unicode letter/mark/number, excluding underscore
_RE_U_WORD = re.compile(r"[^\W_]+", flags=re.UNICODE)


def tokenize_words(text: str, lang: str = "en") -> List[str]:
    s = (text or "").strip()
    if not s:
        return []
    if lang == "ar":
        return _RE_U_WORD.findall(s)
    return _RE_EN_WORD.findall(s)


def _split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _ngrams(tok: List[str], n: int) -> List[Tuple[str, ...]]:
    if len(tok) < n:
        return []
    return [tuple(tok[i : i + n]) for i in range(len(tok) - n + 1)]


def _token_ops(prev_tokens: List[str], cur_tokens: List[str]) -> List[Dict]:
    sm = SequenceMatcher(a=prev_tokens, b=cur_tokens)
    ops: List[Dict] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        # Normalize to the op labels your JS expects
        # JS highlights only "insert" and "replace"
        if tag == "insert":
            op = "insert"
        elif tag == "replace":
            op = "replace"
        elif tag == "delete":
            op = "delete"
        else:
            op = tag

        ops.append(
            {
                "op": op,
                "prev_span": [i1, i2],
                "cur_span": [j1, j2],
                "prev_text": " ".join(prev_tokens[i1:i2]),
                "cur_text": " ".join(cur_tokens[j1:j2]),
            }
        )
    return ops


def build_drift_delta(
    prev_text: str,
    cur_text: str,
    *,
    lang: str = "en",
    top_k_phrases: int = 12,
) -> Dict:
    """
    Builds delta_json payload for Drift underline highlighting.

    IMPORTANT:
    - token indices MUST match the JS tokenizer:
      EN: RE_EN in chat_runtime.js
      AR: RE_U (/[^\W_]+/gu)
    """
    prev_text = (prev_text or "").strip()
    cur_text = (cur_text or "").strip()

    prev_s = _split_sentences(prev_text)
    cur_s = _split_sentences(cur_text)
    prev_set = set(prev_s)
    cur_set = set(cur_s)

    added_sentences = [s for s in cur_s if s not in prev_set]
    removed_sentences = [s for s in prev_s if s not in cur_set]

    prev_t = tokenize_words(prev_text, lang=lang)
    cur_t = tokenize_words(cur_text, lang=lang)

    # phrase stats are best-effort; for AR they’re less meaningful but harmless
    prev_tri = Counter(_ngrams(prev_t, 3))
    cur_tri = Counter(_ngrams(cur_t, 3))

    added_phr = [" ".join(g) for g, _ in (cur_tri - prev_tri).most_common(top_k_phrases)]
    removed_phr = [" ".join(g) for g, _ in (prev_tri - cur_tri).most_common(top_k_phrases)]

    token_ops = _token_ops(prev_t, cur_t)

    return {
        "added_sentences": added_sentences,
        "removed_sentences": removed_sentences,
        "added_phrases": added_phr,
        "removed_phrases": removed_phr,
        "token_ops": token_ops,
    }


# Backwards-compat alias (some older files referenced this name)
def diff_tokens(prev_text: str, cur_text: str, *, lang: str = "en") -> Dict:
    return build_drift_delta(prev_text, cur_text, lang=lang)


def make_recap_en(drift_en: str, *, max_words: int = 28) -> str:
    drift_en = re.sub(r"\s+", " ", (drift_en or "").strip())
    if not drift_en:
        return ""
    first_sent = _split_sentences(drift_en)[:1]
    base = first_sent[0] if first_sent else drift_en
    words = base.split()
    if len(words) <= max_words:
        return base
    return " ".join(words[:max_words]).rstrip() + "…"