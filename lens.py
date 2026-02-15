# lens.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import db
from embeddings_local import local_embed, normalize

# -----------------------------
# Vector lens (existing)
# -----------------------------

PROTOTYPE_SEED_TEXT: Dict[str, str] = {
    "human": "Everyday human time, routine recollection, intimacy, domestic rhythm, personal memory, small gestures.",
    "liminal": "Threshold time, waiting rooms, in-between states, fog, suspension, transit, ambiguity, drift.",
    "environment": "Weather, climate, geography, seasons, air, water, landscape signals, ecological sensing.",
    "environmental": "Weather, climate, geography, seasons, air, water, landscape signals, ecological sensing.",
    "digital": "Network pulses, asynchronous messages, signals, latency, algorithmic time, online fragments.",
    "infrastructure": "Logistics, systems, pipelines, roads, ports, grids, governance mechanisms, maintenance time.",
    "more_than_human": "Deep time, nonhuman agency, ecosystems, migrations, long cycles, planetary rhythm, species-scale.",
}


def weighted_sum(a: List[float], wa: float, b: List[float], wb: float) -> List[float]:
    if len(a) != len(b):
        return a[:] if len(a) >= len(b) else b[:]
    return [(wa * x) + (wb * y) for x, y in zip(a, b)]


def ensure_mind_prototype(conn, *, mind_id: int, mind_key: str, model: str, dims: int) -> Tuple[int, float]:
    seed = PROTOTYPE_SEED_TEXT.get(mind_key, mind_key)
    vec = local_embed(seed, dims=dims)
    vec_blob = __import__("embeddings_local").vector_to_blob(vec)
    vec_hash = __import__("embeddings_local").embedding_hash(model, "prototype", vec_blob)
    emb_id = db.upsert_embedding(conn, kind="prototype", model=model, dims=dims, vector_blob=vec_blob, vector_hash=vec_hash)
    emb_id, w = db.ensure_mind_prototype_row(conn, mind_id=mind_id, version=1, seed_text=seed, embedding_id=emb_id, weight=0.7)
    return emb_id, w


def compute_recent_centroid(conn, *, mind_id: int, dims: int, recent_n: int) -> Optional[List[float]]:
    emb_ids = db.recent_drift_embedding_ids(conn, mind_id=mind_id, n=recent_n)
    if not emb_ids:
        return None

    acc = [0.0] * dims
    count = 0
    for emb_id in emb_ids:
        blob = db.load_embedding_blob(conn, emb_id)
        vec = __import__("embeddings_local").blob_to_vector(blob)
        if len(vec) != dims:
            continue
        for i, x in enumerate(vec):
            acc[i] += x
        count += 1

    if count == 0:
        return None

    centroid = [x / count for x in acc]
    return normalize(centroid)


def build_option3_lens(conn, *, mind: Dict[str, float], model: str, dims: int) -> List[float]:
    proto_emb_id, proto_w = ensure_mind_prototype(conn, mind_id=mind["mind_id"], mind_key=mind["mind_key"], model=model, dims=dims)
    proto_vec = normalize(__import__("embeddings_local").blob_to_vector(db.load_embedding_blob(conn, proto_emb_id)))

    recent_vec = compute_recent_centroid(conn, mind_id=mind["mind_id"], dims=dims, recent_n=int(mind["recent_n"]))
    if recent_vec is None:
        return proto_vec

    lens_vec = weighted_sum(proto_vec, float(proto_w), recent_vec, float(mind["recent_weight"]))
    return normalize(lens_vec)


# -----------------------------
# Concept lens (new, lightweight)
# -----------------------------

# Memory strata concept:
# - hard_invariants persist forever (sensorial/affective cues)
# - soft_invariants blur slowly (town > alley-name, time-of-day persists but precision fades)
# - volatile_details blur freely (names, exact labels)
#
# This is used as a contract for prompt + enforcement (pipeline/openai).
DECAY_POLICIES: Dict[str, Dict[str, List[str]]] = {
    "human": {
        "hard_invariants": ["emotion", "bodily_sensation", "atmosphere", "audiovisual_texture"],
        "soft_invariants": ["time_of_day", "spatial_scale", "town_level_place"],
        "volatile_details": ["names", "alley_names", "exact_labels"],
    },
    "liminal": {
        "hard_invariants": ["ambiguity_feel", "threshold_sense", "atmosphere", "audiovisual_texture"],
        "soft_invariants": ["time_of_day", "spatial_scale"],
        "volatile_details": ["names", "precise_boundaries"],
    },
    "environment": {
        "hard_invariants": ["weather_feel", "material_conditions", "air_water_light", "audiovisual_texture"],
        "soft_invariants": ["seasonal_phase", "landscape_scale"],
        "volatile_details": ["human_names", "exact_dates"],
    },
    "digital": {
        "hard_invariants": ["signal_feel", "latency_rhythm", "fragmentation_texture"],
        "soft_invariants": ["network_context", "platform_shape"],
        "volatile_details": ["exact_handles", "precise_sources"],
    },
    "infrastructure": {
        "hard_invariants": ["pathway_logic", "circulation_pressure", "maintenance_texture"],
        "soft_invariants": ["spatial_scale", "time_of_day"],
        "volatile_details": ["facility_names", "exact_routes"],
    },
    "more_than_human": {
        "hard_invariants": ["nonhuman_attunement", "cycle_feel", "planetary_rhythm"],
        "soft_invariants": ["seasonal_phase", "ecological_scale"],
        "volatile_details": ["human_labels", "exact_places"],
    },
}

# alias
DECAY_POLICIES["environmental"] = DECAY_POLICIES["environment"]


def decay_policy_for(mind_key: str) -> Dict[str, List[str]]:
    return DECAY_POLICIES.get(mind_key, DECAY_POLICIES["human"])


# ------------------------------------------------
# Per-mind resonance keywords for headline selection
# These define what each temporality "notices" in the news.
# Headlines are scored by keyword overlap + always
# get a small base score so no mind is starved of input.
# ------------------------------------------------

import re as _re

RESONANCE_KEYWORDS: Dict[str, List[str]] = {
    "human": [
        "family", "home", "child", "parent", "school", "neighborhood", "community",
        "grief", "joy", "love", "loss", "memory", "routine", "morning", "evening",
        "kitchen", "garden", "friend", "illness", "health", "birth", "death",
        "story", "personal", "daily", "local", "people", "worker", "care",
        "food", "meal", "sleep", "body", "feeling", "emotion", "heart",
    ],
    "liminal": [
        "border", "threshold", "transit", "waiting", "between", "uncertain",
        "ambiguous", "fog", "twilight", "dawn", "dusk", "limbo", "suspended",
        "transition", "refugee", "migration", "crossing", "passage", "interim",
        "neither", "both", "almost", "edge", "margin", "unknown", "shift",
        "change", "crisis", "turning", "moment", "pause", "stillness",
    ],
    "environment": [
        "climate", "weather", "storm", "rain", "drought", "flood", "wildfire",
        "temperature", "ocean", "river", "forest", "soil", "air", "water",
        "pollution", "emission", "carbon", "ice", "glacier", "sea level",
        "wind", "season", "heat", "cold", "landscape", "erosion", "earth",
        "atmosphere", "ozone", "ecosystem", "habitat", "coral", "arctic",
    ],
    "digital": [
        "AI", "algorithm", "data", "cyber", "hack", "network", "internet",
        "social media", "platform", "app", "software", "tech", "digital",
        "online", "virtual", "crypto", "blockchain", "surveillance", "privacy",
        "signal", "bandwidth", "server", "cloud", "streaming", "bot",
        "deepfake", "automation", "chip", "silicon", "compute", "model",
    ],
    "infrastructure": [
        "road", "bridge", "pipeline", "grid", "power", "energy", "rail",
        "port", "airport", "traffic", "supply chain", "logistics", "freight",
        "construction", "maintenance", "utility", "water system", "sewage",
        "highway", "tunnel", "dam", "electricity", "oil", "gas", "transit",
        "shipping", "cargo", "route", "corridor", "facility", "plant",
    ],
    "more_than_human": [
        "species", "extinction", "biodiversity", "migration", "evolution",
        "ecosystem", "ocean", "forest", "animal", "bird", "whale", "insect",
        "coral", "reef", "polar", "arctic", "antarctic", "ancient", "deep time",
        "geological", "volcano", "earthquake", "planetary", "asteroid",
        "moss", "lichen", "fungal", "mycelium", "tree", "root", "soil",
        "nonhuman", "organism", "bacteria", "marine", "wilderness",
    ],
}
RESONANCE_KEYWORDS["environmental"] = RESONANCE_KEYWORDS["environment"]


def _score_headline_for_mind(headline: str, mind_key: str) -> float:
    """
    Score a headline's resonance with a specific temporality mind.
    Higher = more resonant. Every headline gets a small base score
    so no mind is ever fully starved of input.
    Uses word-boundary matching for short keywords to avoid false positives
    (e.g., "AI" matching inside "air quality").
    """
    keywords = RESONANCE_KEYWORDS.get(mind_key, RESONANCE_KEYWORDS["human"])
    h_lower = headline.lower()
    score = 0.1  # base score — every headline has slight relevance
    for kw in keywords:
        kw_lower = kw.lower()
        # For short keywords (<=3 chars), use word boundary regex to avoid substring false positives
        if len(kw_lower) <= 3:
            if _re.search(r"\b" + _re.escape(kw_lower) + r"\b", h_lower):
                score += 1.0 + (0.1 * len(kw.split()))
        else:
            if kw_lower in h_lower:
                score += 1.0 + (0.1 * len(kw.split()))
    return score


def select_headlines_for_mind(
    headlines: List[str],
    mind_key: str,
    max_headlines: int = 6,
    min_headlines: int = 2,
) -> Dict[str, Any]:
    """
    Select the most resonant headlines for a specific temporality mind.

    Returns a dict with:
      - "headlines": list of selected headline strings
      - "resonance": float 0.0-1.0 indicating how well headlines matched this mind
        Low resonance (< 0.3) means the present moment doesn't connect to this
        temporality's lens — the mind should hallucinate (semantic instability).
      - "resonant_count": how many headlines had keyword hits
    """
    if not headlines:
        return {"headlines": [], "resonance": 0.0, "resonant_count": 0}

    scored = [(h, _score_headline_for_mind(h, mind_key)) for h in headlines]
    scored.sort(key=lambda x: -x[1])

    # Take headlines that scored above base (had keyword hits)
    resonant = [(h, s) for h, s in scored if s > 0.15]
    resonant_count = len(resonant)

    # Resonance: ratio of headlines that actually matched keywords
    # Clamped to 0.0-1.0
    resonance = min(1.0, resonant_count / max(min_headlines * 2, 1))

    if resonant_count >= min_headlines:
        selected = [h for h, _ in resonant[:max_headlines]]
    else:
        # Fall back to top N by score even if no keywords matched
        selected = [h for h, _ in scored[:max(min_headlines, 1)]]

    return {
        "headlines": selected,
        "resonance": round(resonance, 2),
        "resonant_count": resonant_count,
    }