# embeddings_local.py
from __future__ import annotations

import hashlib
import json
import math
from typing import List


DEFAULT_EMBED_DIMS = 256


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def local_embed(text: str, dims: int = DEFAULT_EMBED_DIMS) -> List[float]:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    vec: List[float] = []
    seed = h
    while len(vec) < dims:
        seed = hashlib.sha256(seed).digest()
        for b in seed:
            vec.append((b / 127.5) - 1.0)
            if len(vec) >= dims:
                break
    return vec[:dims]


def normalize(v: List[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in v))
    if n <= 1e-12:
        return v[:]
    return [x / n for x in v]


def cosine_sim(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return dot / math.sqrt(na * nb)


def vector_to_blob(vec: List[float]) -> bytes:
    return json.dumps(vec, ensure_ascii=False).encode("utf-8")


def blob_to_vector(blob: bytes) -> List[float]:
    return json.loads(blob.decode("utf-8"))


def embedding_hash(model: str, kind: str, vec_blob: bytes) -> str:
    return sha256_text(f"{model}:{kind}:{vec_blob.decode('utf-8')}")