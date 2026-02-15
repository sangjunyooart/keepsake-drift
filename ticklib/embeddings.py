# ticklib/embeddings.py
from __future__ import annotations

import hashlib
from typing import Iterable, List, Sequence

from .embeddings_local import local_embed_texts
from . import db


def blob_hash(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def embedding_hash(model: str, kind: str, vector_blob: bytes) -> str:
    # stable hash key used for upsert de-dupe
    return hashlib.sha256((model + ":" + kind + ":" + blob_hash(vector_blob)).encode("utf-8")).hexdigest()


def _vec_to_blob(vec: Sequence[float]) -> bytes:
    # Store as comma-separated floats (simple, stable, and DB-friendly)
    # IMPORTANT: keep consistent with existing DB expectations in db.upsert_embedding().
    return (",".join(f"{float(x):.8f}" for x in vec)).encode("utf-8")


# --------------------------------------------------------------------
# Public API expected by pipeline / other modules
# --------------------------------------------------------------------

def embed_text(text: str, *, model: str, dims: int) -> List[float]:
    """
    Back-compat API:
      pipeline.py (or other code) expects `embed_text` to exist in ticklib.embeddings.

    Returns the embedding vector for a single string.
    """
    vecs = local_embed_texts([text], model=model, dims=dims)
    return list(vecs[0]) if vecs else []


def embed_texts(texts: Sequence[str], *, model: str, dims: int) -> List[List[float]]:
    """
    Convenience wrapper for batching.
    """
    vecs = local_embed_texts(list(texts), model=model, dims=dims)
    return [list(v) for v in vecs]


def ensure_event_embeddings(conn, event_ids: Sequence[int], events_by_id: dict, *, model: str, dims: int):
    """
    Ensures embeddings exist for the given event_ids.
    Returns: dict[event_id] -> embedding_id
    """
    out = {}
    missing_texts = []
    missing_ids = []

    for eid in event_ids:
        emb_id = db.get_event_embedding_id(conn, eid, model=model, dims=dims)
        if emb_id:
            out[eid] = emb_id
            continue
        text = events_by_id.get(eid, "") or ""
        missing_ids.append(eid)
        missing_texts.append(text)

    if missing_ids:
        vecs = local_embed_texts(missing_texts, model=model, dims=dims)
        for eid, vec in zip(missing_ids, vecs):
            vec_blob = _vec_to_blob(vec)
            vec_hash = embedding_hash(model, "event", vec_blob)
            emb_id = db.upsert_embedding(
                conn,
                kind="event",
                model=model,
                dims=dims,
                vector_blob=vec_blob,
                vector_hash=vec_hash,
            )
            db.set_event_embedding_id(conn, eid, emb_id, model=model, dims=dims)
            out[eid] = emb_id

    return out


def ensure_drift_embedding(conn, drift_text: str, *, model: str, dims: int):
    """
    Creates/updates an embedding row for the drift text and returns embedding_id.
    The caller is responsible for linking it to drift_memory.embedding_id if needed.
    """
    vec = embed_text(drift_text or "", model=model, dims=dims)
    vec_blob = _vec_to_blob(vec)
    vec_hash = embedding_hash(model, "drift_memory", vec_blob)
    return db.upsert_embedding(
        conn,
        kind="drift_memory",
        model=model,
        dims=dims,
        vector_blob=vec_blob,
        vector_hash=vec_hash,
    )