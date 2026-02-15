# app.py
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config
import engine
import safety
import storage

from openai import OpenAI

# ---- delta import (supports ticklib/delta.py without requiring a package) ----
try:
    import delta  # type: ignore
except ModuleNotFoundError:
    import importlib.util

    _HERE = Path(__file__).resolve().parent
    _DELTA_PATH = _HERE / "ticklib" / "delta.py"

    if not _DELTA_PATH.exists():
        raise

    spec = importlib.util.spec_from_file_location("delta", str(_DELTA_PATH))
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"Could not load delta from {_DELTA_PATH}")

    delta = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(delta)  # type: ignore
# ---------------------------------------------------------------------------

ENABLE_ARABIC = os.getenv("ENABLE_ARABIC", "1").strip().lower() in ("1", "true", "yes", "y")
ENABLE_TRANSLATION = ENABLE_ARABIC  # alias
TRANSLATE_MODEL = os.getenv("OPENAI_TRANSLATE_MODEL", "gpt-4o-mini")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

INDEX_HTML = STATIC_DIR / "index.html"
CHAT_HTML = STATIC_DIR / "chat.html"
PERSONA_HTML = STATIC_DIR / "persona.html"
PRIVACY_HTML = STATIC_DIR / "privacy.html"
CHAT_MOD_HTML = STATIC_DIR / "chat_mod.html"  # optional

app = FastAPI()

# CORS middleware for Cloudflare Pages frontend
# Use regex pattern to allow all Cloudflare Pages domains (with or without subdomain hash)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://([a-zA-Z0-9-]+\.)?keepsake-drift\.pages\.dev",
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods including OPTIONS
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

# Add explicit OPTIONS handler for CORS preflight (FastAPI/Starlette doesn't auto-handle OPTIONS)
@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    return {}

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Serve generated drift images from data/images/
DATA_IMAGES_DIR = BASE_DIR / "data" / "images"
DATA_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/drift_images", StaticFiles(directory=str(DATA_IMAGES_DIR)), name="drift_images")

_client: Optional[OpenAI] = None


_tick_log = logging.getLogger("tick_loop")

# ── Global tick cycle ──────────────────────────────────────────────
# TICK_DISPLAY_INTERVAL: how often the frontend shows a new drift (seconds)
# TICK_LEAD_TIME: how many seconds BEFORE display to start processing
#   (ingest RSS + run drift pipeline).  Processing takes ~3 min,
#   so a 5-min lead gives comfortable headroom.
TICK_DISPLAY_INTERVAL = int(os.getenv("KD_TICK_INTERVAL", "1800"))   # 30 min
TICK_LEAD_TIME = int(os.getenv("KD_TICK_LEAD", "300"))              # 5 min before
TICK_ENABLED = os.getenv("KD_TICK_ENABLED", "1").strip().lower() in ("1", "true", "yes")

_tick_thread: Optional[threading.Thread] = None
_tick_stop = threading.Event()


def _run_one_tick() -> None:
    """Execute one full cycle: RSS ingest → drift pipeline."""
    db_path = getattr(config, "SQLITE_PATH", "./data/keepsake.sqlite")
    try:
        # Phase 1: RSS ingest
        _tick_log.info("Tick loop: starting RSS ingest …")
        from rss_ingest_orchestrator import run_ingest
        ingest_result = run_ingest(
            db_path=db_path,
            mode="random",
            sources_per_run=6,
            event_limit=60,
            seed=None,
            timeout=20,
        )
        tick_id = ingest_result.get("tick_id")
        event_ids = ingest_result.get("inserted_event_ids") or []
        _tick_log.info(
            "Tick loop: ingest done — tick_id=%s, events=%d",
            tick_id, len(event_ids),
        )

        if tick_id is None:
            _tick_log.warning("Tick loop: ingest returned no tick_id, skipping drift.")
            return

        # Phase 2: Drift pipeline (all 6 minds + image gen)
        _tick_log.info("Tick loop: starting drift pipeline …")
        from ticklib.pipeline import run_tick
        result = run_tick(
            db_path=db_path,
            tick_id=int(tick_id),
            event_ids=event_ids,
            timeout_seconds=180.0,
        )
        _tick_log.info("Tick loop: drift complete — %s", result.get("status", "?"))

    except Exception:
        _tick_log.exception("Tick loop: error during tick execution")


def _tick_loop() -> None:
    """
    Background thread that runs ticks on a fixed schedule.

    Timeline for each 10-minute window:
        0:00  — frontend countdown starts (display epoch)
        5:00  — processing begins (TICK_LEAD_TIME before next display)
       ~8:00  — processing finishes
       10:00  — frontend countdown resets; new drift visible via /state poll
    """
    _tick_log.info(
        "Tick loop started — display every %ds, processing starts %ds before display.",
        TICK_DISPLAY_INTERVAL, TICK_LEAD_TIME,
    )

    # Align to the next display boundary
    step = TICK_DISPLAY_INTERVAL
    now = time.time()
    next_display = (now // step + 1) * step  # next clean boundary
    next_process = next_display - TICK_LEAD_TIME

    # If we already passed the processing point for the upcoming boundary, skip to the one after
    if next_process < now:
        next_display += step
        next_process = next_display - TICK_LEAD_TIME

    while not _tick_stop.is_set():
        now = time.time()
        sleep_for = next_process - now
        if sleep_for > 0:
            _tick_stop.wait(timeout=sleep_for)
            if _tick_stop.is_set():
                break

        _tick_log.info(
            "Tick loop: processing now (display at +%ds)",
            TICK_LEAD_TIME,
        )
        _run_one_tick()

        # Advance to next cycle
        next_display += step
        next_process = next_display - TICK_LEAD_TIME


@app.on_event("startup")
def _startup():
    storage.init_db()

    # Launch background tick loop
    global _tick_thread
    if TICK_ENABLED and _tick_thread is None:
        _tick_thread = threading.Thread(target=_tick_loop, daemon=True, name="tick-loop")
        _tick_thread.start()
        _tick_log.info("Background tick thread launched.")


@app.on_event("shutdown")
def _shutdown():
    _tick_stop.set()
    if _tick_thread is not None:
        _tick_thread.join(timeout=5)


def _serve_html(path: Path) -> FileResponse:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Missing HTML file: {path}")
    return FileResponse(str(path), media_type="text/html")


def _client_or_none() -> Optional[OpenAI]:
    global _client
    if _client is not None:
        return _client
    if not os.getenv("OPENAI_API_KEY"):
        return None
    _client = OpenAI()
    return _client


def _table_columns(conn: sqlite3.Connection, table: str) -> Set[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
        # sqlite3.Row might be enabled; handle both tuple and mapping
        cols: Set[str] = set()
        for r in rows:
            try:
                cols.add(r["name"])
            except Exception:
                cols.add(r[1])
        return cols
    except Exception:
        return set()


def _translate_cached(
    src_text: str,
    *,
    src_lang: str,
    tgt_lang: str,
    db_path: str,
    allow_remote: bool = True,
) -> str:
    src_text = (src_text or "").strip()
    if not src_text:
        return ""
    if src_lang == tgt_lang:
        return src_text

    cached = storage.get_cached_translation(src_lang, tgt_lang, src_text, db_path=db_path)
    if cached:
        return cached

    if not allow_remote:
        return ""

    client = _client_or_none()
    if client is None:
        return ""

    prompt = (
        f"Translate the following text from {src_lang} to {tgt_lang}. "
        f"Return only the translation. Preserve paragraph breaks.\n\n{src_text}"
    )

    try:
        resp = client.responses.create(
            model=TRANSLATE_MODEL,
            input=prompt,
            timeout=30.0,
        )
        out = (resp.output_text or "").strip()
    except Exception:
        out = ""

    if out:
        storage.put_cached_translation(src_lang, tgt_lang, src_text, out, db_path=db_path)
    return out


def _call_generate_chat_reply(
    *,
    temporality: str,
    user_text: str,
    user_lang: str,
    active_drift_en: str,
    active_drift_ar: str,
    drift_context: str = "",
    session_id: str,
) -> Dict[str, Any]:
    fn = getattr(engine, "generate_chat_reply", None)
    if fn is None:
        return {"reply_en": "Engine missing generate_chat_reply.", "reply_ar": ""}

    sig = inspect.signature(fn)
    params = set(sig.parameters.keys())

    kwargs: Dict[str, Any] = {}

    if "temporality" in params:
        kwargs["temporality"] = temporality
    elif "persona" in params:
        kwargs["persona"] = temporality

    if "user_text" in params:
        kwargs["user_text"] = user_text
    elif "message" in params:
        kwargs["message"] = user_text
    elif "text" in params:
        kwargs["text"] = user_text
    elif "prompt" in params:
        kwargs["prompt"] = user_text

    if "user_lang" in params:
        kwargs["user_lang"] = user_lang
    elif "lang" in params:
        kwargs["lang"] = user_lang
    elif "language" in params:
        kwargs["language"] = user_lang

    if "active_drift_en" in params:
        kwargs["active_drift_en"] = active_drift_en
    if "active_drift_ar" in params:
        kwargs["active_drift_ar"] = active_drift_ar
    if "drift_en" in params and "active_drift_en" not in params:
        kwargs["drift_en"] = active_drift_en
    if "drift_ar" in params and "active_drift_ar" not in params:
        kwargs["drift_ar"] = active_drift_ar

    if "drift_context" in params:
        kwargs["drift_context"] = drift_context

    if "session_id" in params:
        kwargs["session_id"] = session_id
    elif "session" in params:
        kwargs["session"] = session_id

    try:
        out = fn(**kwargs)  # type: ignore[misc]
        if isinstance(out, dict):
            return out
        return {"reply_en": str(out), "reply_ar": ""}
    except Exception as e:
        return {"reply_en": f"Chat error: {type(e).__name__}: {e}", "reply_ar": ""}


@app.get("/tick_status")
async def tick_status():
    """Return tick cycle info so the frontend can align its countdown."""
    now = time.time()
    step = TICK_DISPLAY_INTERVAL
    next_display = (now // step + 1) * step
    return JSONResponse({
        "tick_interval_s": step,
        "lead_time_s": TICK_LEAD_TIME,
        "next_display_epoch_ms": int(next_display * 1000),
        "server_now_ms": int(now * 1000),
        "enabled": TICK_ENABLED,
    })


@app.get("/lang_config")
async def lang_config_endpoint():
    """Expose active second language config to the frontend."""
    lc = config.LANG_CONFIG.get(config.SECOND_LANG, config.LANG_CONFIG["ar"])
    return JSONResponse({
        "second_lang": config.SECOND_LANG,
        "lang_name": lc["name"],
        "native_name": lc.get("native_name", ""),
        "toggle_label": lc.get("toggle_label", ""),
        "direction": lc["direction"],
        "enable_translation": ENABLE_TRANSLATION,
    })


@app.get("/tunnel_url")
async def get_tunnel_url():
    """Return the current Cloudflare tunnel URL for dynamic frontend configuration."""
    import os
    tunnel_url_file = os.path.join(os.path.dirname(__file__), "data", "tunnel_url.txt")

    if os.path.exists(tunnel_url_file):
        with open(tunnel_url_file, 'r') as f:
            tunnel_url = f.read().strip()
            return JSONResponse({"tunnel_url": tunnel_url})
    else:
        return JSONResponse({"tunnel_url": None, "error": "Tunnel URL not found"}, status_code=404)


@app.get("/")
async def index():
    return _serve_html(INDEX_HTML)


@app.get("/chat")
async def chat():
    return _serve_html(CHAT_HTML)


@app.get("/chat_mod")
async def chat_mod():
    if CHAT_MOD_HTML.exists():
        return _serve_html(CHAT_MOD_HTML)
    return JSONResponse({"error": "Missing chat_mod.html"}, status_code=404)


@app.get("/persona")
async def persona():
    if PERSONA_HTML.exists():
        return _serve_html(PERSONA_HTML)
    return JSONResponse({"ok": True})


@app.get("/privacy")
async def privacy():
    if PRIVACY_HTML.exists():
        return _serve_html(PRIVACY_HTML)
    return JSONResponse({"ok": True})


# Museum stream pages (portrait 9:16 display for installation)
MUSEUM_HUMAN_HTML = STATIC_DIR / "museum_human.html"
MUSEUM_LIMINAL_HTML = STATIC_DIR / "museum_liminal.html"
MUSEUM_ENV_HTML = STATIC_DIR / "museum_environment.html"

@app.get("/museum/human")
async def museum_human():
    if not MUSEUM_HUMAN_HTML.exists():
        raise HTTPException(status_code=404, detail="Museum human stream not found")
    return FileResponse(MUSEUM_HUMAN_HTML)

@app.get("/museum/liminal")
async def museum_liminal():
    if not MUSEUM_LIMINAL_HTML.exists():
        raise HTTPException(status_code=404, detail="Museum liminal stream not found")
    return FileResponse(MUSEUM_LIMINAL_HTML)

@app.get("/museum/environment")
async def museum_environment():
    if not MUSEUM_ENV_HTML.exists():
        raise HTTPException(status_code=404, detail="Museum environment stream not found")
    return FileResponse(MUSEUM_ENV_HTML)


@app.post("/chat_ui")
async def chat_ui(payload: Dict[str, Any]):
    db_path = getattr(config, "SQLITE_PATH", "./data/keepsake.sqlite")

    session_id = (payload.get("session_id") or "").strip()
    temporality = (payload.get("persona") or payload.get("temporality") or "").strip()
    user_text = (payload.get("message") or payload.get("text") or "").strip()
    user_lang = (payload.get("lang") or "en").strip()

    if temporality not in config.TEMPORALITIES:
        return JSONResponse({"error": "Unknown persona", "supported": config.TEMPORALITIES}, status_code=400)
    if not user_text:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    # ── Moderation: block profanity, religious offense, flagged content ──
    client = _client_or_none()
    if client:
        cleaned, meta = safety.preprocess_inbound_text(
            client, user_text, session_id=session_id, purpose="chat"
        )
        if meta.get("blocked"):
            block_msg = "This space holds memory gently. Let\u2019s keep it that way."
            return JSONResponse({"reply_en": block_msg, "reply_ar": block_msg})
        if cleaned:
            user_text = cleaned

    latest = _get_latest_drift_full(temporality, db_path=db_path)
    active_en = (latest.get("drift_text") if latest else "") or ""

    active_ar = ""
    if ENABLE_ARABIC and active_en:
        active_ar = (latest.get("drift_text_ar") or "").strip()
        if not active_ar:
            active_ar = _translate_cached(active_en, src_lang="en", tgt_lang=config.SECOND_LANG, db_path=db_path, allow_remote=True)

    # Extract drift context for richer chatbot persona
    drift_context = ""
    if latest:
        pj_raw = (latest.get("params_json") or "").strip()
        keepsake = (latest.get("keepsake_text") or "").strip()
        if pj_raw:
            try:
                pj = json.loads(pj_raw)
                direction = (pj.get("drift_direction") or "").strip()
                justification = (pj.get("justification_en") or "").strip()
                parts = []
                if direction:
                    parts.append(f"What I perceive in the present moment: {direction}")
                if justification:
                    parts.append(f"Why my memory shifted: {justification}")
                if keepsake:
                    parts.append(f"How I have been changing: {keepsake}")
                drift_context = "\n".join(parts)
            except (json.JSONDecodeError, TypeError):
                if keepsake:
                    drift_context = f"How I have been changing: {keepsake}"

    result = _call_generate_chat_reply(
        temporality=temporality,
        user_text=user_text,
        user_lang=user_lang,
        active_drift_en=active_en,
        active_drift_ar=active_ar,
        drift_context=drift_context,
        session_id=session_id,
    )

    reply_en = (result.get("reply_en") or "").strip()
    reply_ar = (result.get("reply_ar") or "").strip()

    if not ENABLE_ARABIC:
        reply_ar = ""

    return JSONResponse({"reply_en": reply_en, "reply_ar": reply_ar})


@app.get("/keepsake_archive")
async def keepsake_archive(persona: str = "liminal"):
    """
    Return all past keepsake texts + visitor fragments for a given persona.
    Ordered oldest-first (chronological). Each entry has a `source` field:
    "drift" for AI keepsakes, "visitor" for user fragments.
    """
    persona = (persona or "").strip()
    if persona not in config.TEMPORALITIES:
        return JSONResponse({"error": "Unknown persona", "supported": config.TEMPORALITIES}, status_code=400)

    db_path = getattr(config, "SQLITE_PATH", "./data/keepsake.sqlite")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        entries = []

        # 1) Drift keepsake entries
        cols = _table_columns(conn, "drift_memory")
        if "keepsake_text" in cols:
            mind_id = storage.mind_id_for_temporality(conn, persona)
            has_params = "params_json" in cols

            select_cols = ["version", "keepsake_text", "created_at"]
            if has_params:
                select_cols.append("params_json")

            rows = conn.execute(
                f"""
                SELECT {", ".join(select_cols)}
                FROM drift_memory
                WHERE mind_id = ?
                  AND keepsake_text IS NOT NULL
                  AND length(trim(keepsake_text)) > 0
                ORDER BY version ASC;
                """,
                (mind_id,),
            ).fetchall()

            for r in rows:
                entry = {
                    "source": "drift",
                    "version": int(r["version"]),
                    "keepsake_text": (r["keepsake_text"] or "").strip(),
                    "keepsake_text_ar": "",
                    "created_at": (r["created_at"] or "").strip(),
                    "curated_headlines": [],
                }
                if has_params:
                    raw_pj = (r["params_json"] or "").strip()
                    if raw_pj:
                        try:
                            pj = json.loads(raw_pj)
                            entry["curated_headlines"] = pj.get("curated_headlines") or []
                            entry["keepsake_text_ar"] = (pj.get("keepsake_text_ar") or "").strip()
                        except (json.JSONDecodeError, TypeError):
                            pass
                entries.append(entry)

        # 2) Visitor fragment entries
        frag_cols = _table_columns(conn, "user_fragments")
        if frag_cols:  # table exists
            has_text_ar = "text_ar" in frag_cols
            frag_select = "text, created_at"
            if has_text_ar:
                frag_select = "text, text_ar, created_at"

            frag_rows = conn.execute(
                f"""
                SELECT {frag_select}
                FROM user_fragments
                WHERE persona = ?
                ORDER BY created_at ASC;
                """,
                (persona,),
            ).fetchall()

            for fr in frag_rows:
                entries.append({
                    "source": "visitor",
                    "version": None,
                    "keepsake_text": (fr["text"] or "").strip(),
                    "keepsake_text_ar": (fr["text_ar"] or "").strip() if has_text_ar else "",
                    "created_at": (fr["created_at"] or "").strip(),
                    "curated_headlines": [],
                })

        # Sort all entries by created_at (oldest first)
        def sort_key(e):
            ts = e.get("created_at") or ""
            return ts
        entries.sort(key=sort_key)

        return JSONResponse({"entries": entries})
    finally:
        conn.close()


FRAGMENT_REFRAME_MODEL = os.getenv("OPENAI_FRAGMENT_MODEL", "gpt-4o-mini")

# Words/phrases that are conversational acknowledgments, not memory fragments.
# Includes single words AND multi-word casual phrases.
_TRIVIAL_PHRASES: set[str] = {
    # English — single words
    "yes", "no", "ok", "okay", "sure", "yeah", "yep", "yea", "ya", "nah",
    "nope", "thanks", "thank you", "cool", "nice", "great", "fine", "good",
    "right", "alright", "hmm", "hm", "true", "false", "lol", "haha", "wow",
    "k", "thx", "ty", "hi", "hello", "hey", "bye", "sup", "yo", "yay",
    "damn", "dope", "sick", "lit", "word", "bet", "bruh", "bro", "dude",
    "same", "mood", "omg", "idk", "nvm", "imo", "tbh", "smh", "wtf",
    # English — multi-word casual/greetings
    "what's up", "whats up", "wassup", "wazzup", "sup bro",
    "how are you", "how r u", "how you doing", "how's it going",
    "what's going on", "whats going on", "what's new", "whats new",
    "good morning", "good night", "good evening", "good afternoon",
    "thank you so much", "thanks a lot", "thanks man", "thank u",
    "i see", "i know", "got it", "i agree", "me too", "same here",
    "no way", "for real", "oh well", "oh ok", "oh okay", "oh really",
    "not really", "not sure", "no thanks", "no thank you",
    "sounds good", "all good", "looks good", "that's cool", "thats cool",
    "that's nice", "thats nice", "that's great", "thats great",
    "i guess", "i think so", "i don't know", "i dont know",
    "see you", "see ya", "take care", "have a good one",
    "nice one", "well done", "good job", "my bad", "no worries",
    "of course", "why not", "let's go", "lets go", "go ahead",
    "haha nice", "lol ok", "ok cool", "yeah sure", "yes please",
    "no problem", "np",
    # Arabic — single words
    "أجل", "نعم", "لا", "شكراً", "شكرا", "حسناً", "حسنا", "مرحبا", "طيب",
    "تمام", "ممتاز", "صح", "أوكي", "يب", "لأ", "اهلا", "باي", "يلا",
    # Arabic — multi-word casual
    "كيف حالك", "كيفك", "شو أخبارك", "شلونك", "ايش الأخبار",
    "صباح الخير", "مساء الخير", "تصبح على خير",
    "الله يعطيك العافية", "يعطيك العافية", "ما شاء الله",
    "إن شاء الله", "ان شاء الله",
    # Greek — single words
    "ναι", "όχι", "οχι", "εντάξει", "ενταξει", "ευχαριστώ", "ευχαριστω",
    "ωραία", "ωραια", "καλά", "καλα", "σωστά", "μπράβο", "γεια", "φυσικά",
    "σούπερ", "τέλεια", "τελεια", "πάμε",
    # Greek — multi-word casual
    "γεια σου", "γεια σας", "τι κάνεις", "τι κανεις", "καλημέρα",
    "καλησπέρα", "καληνύχτα", "πώς είσαι", "πως εισαι",
    "τι γίνεται", "τι λες", "όλα καλά", "ολα καλα", "δεν ξέρω",
    "δεν πειράζει", "ευχαριστώ πολύ", "τα λέμε",
    # Brazilian Portuguese — single words
    "sim", "não", "nao", "obrigado", "obrigada", "legal", "beleza",
    "valeu", "verdade", "falou", "massa", "opa", "eita", "tchau",
    "oi", "olá", "ola", "vlw", "blz", "tmj",
    # Brazilian Portuguese — multi-word casual
    "tudo bem", "tudo bom", "como vai", "bom dia", "boa tarde",
    "boa noite", "muito obrigado", "muito obrigada", "de nada",
    "tá bom", "ta bom", "sei lá", "sei la", "até mais", "ate mais",
    "que legal", "que massa", "é isso", "pois é", "pode ser",
    "tudo certo", "sem problemas",
}


def _is_trivial_input(text: str) -> bool:
    """Return True if text is a short conversational phrase, not a meaningful memory fragment."""
    t = (text or "").strip()
    if len(t) <= 2:
        return True  # single char / emoji
    words = t.split()
    if len(words) > 5:
        return False  # 6+ words = likely meaningful enough
    normalized = t.lower().rstrip(".!?,;:؟،")
    # Normalize curly/smart quotes to straight ASCII (browser input varies)
    normalized = normalized.replace("\u2018", "'").replace("\u2019", "'")
    normalized = normalized.replace("\u201c", '"').replace("\u201d", '"')
    # Check exact match against known trivial phrases
    if normalized in _TRIVIAL_PHRASES:
        return True
    # Also strip common leading fillers and re-check
    for prefix in ("so ", "well ", "um ", "uh ", "like ", "hey ", "oh "):
        if normalized.startswith(prefix):
            remainder = normalized[len(prefix):].strip()
            if remainder in _TRIVIAL_PHRASES:
                return True
    return False


# Fragment reframe cache: {sha256(persona:text:context): (reframed_text, timestamp)}
# TTL: 1 hour, max 500 entries
_reframe_cache: Dict[str, Tuple[str, float]] = {}


@app.post("/leave_fragment")
async def leave_fragment(payload: Dict[str, Any]):
    """
    Accept a user memory fragment. Moderated via safety pipeline.
    AI interprets it into a single poetic sentence, then stores it
    as a Visitor archive entry.
    """
    db_path = getattr(config, "SQLITE_PATH", "./data/keepsake.sqlite")

    session_id = (payload.get("session_id") or "").strip()
    persona = (payload.get("persona") or "").strip()
    text = (payload.get("text") or "").strip()

    if persona not in config.TEMPORALITIES:
        return JSONResponse({"ok": False, "message": "Unknown persona"}, status_code=400)
    if not text:
        return JSONResponse({"ok": False, "message": "Empty fragment"}, status_code=400)

    # Short conversational words (yes, no, ok, thanks...) — acknowledge but don't archive
    if _is_trivial_input(text):
        return JSONResponse({"ok": False, "trivial": True, "message": "Noted."})

    # Moderation
    client = _client_or_none()
    if client:
        cleaned, meta = safety.preprocess_inbound_text(
            client, text, session_id=session_id, purpose="fragment"
        )
        if meta.get("blocked"):
            return JSONResponse({
                "ok": False,
                "message": "This space holds memory gently. Let\u2019s keep it that way."
            })
        if cleaned:
            text = cleaned

    # The temporality hears what the visitor said and writes it in its own terms.
    # Fetch current drift context so the AI has its voice.
    drift_voice = ""
    try:
        latest = _get_latest_drift_full(persona, db_path=db_path)
        if latest:
            pj_raw = (latest.get("params_json") or "").strip()
            keepsake = (latest.get("keepsake_text") or "").strip()
            drift_txt = (latest.get("drift_text") or "").strip()[:400]
            if pj_raw:
                try:
                    pj = json.loads(pj_raw)
                    direction = (pj.get("drift_direction") or "").strip()
                    if direction:
                        drift_voice += f"What you are perceiving now: {direction}\n"
                except (json.JSONDecodeError, TypeError):
                    pass
            if drift_txt:
                drift_voice += f"Your current state of memory:\n{drift_txt}\n"
            if keepsake:
                drift_voice += f"How you have been changing:\n{keepsake[:300]}\n"
    except Exception:
        pass

    reframed = text  # fallback: use cleaned text as-is

    # Check reframe cache (1-hour TTL)
    cache_key = hashlib.sha256(f"{persona}:{text}:{drift_voice[:200]}".encode()).hexdigest()
    now = time.time()

    if cache_key in _reframe_cache:
        cached_reframe, timestamp = _reframe_cache[cache_key]
        if now - timestamp < 3600:  # 1 hour = 3600 seconds
            reframed = cached_reframe
            client = None  # Skip API call entirely

    if client:
        try:
            system_content = (
                f"You are the {persona} perceiver — a machinic observer of time. "
                f"An unknown visitor just passed through and left a fragment of words. "
                f"You heard what they said. Now write it down in your own terms, "
                f"as you understood it — filtered through your way of perceiving. "
                f"Begin with something like 'A visitor mentioned...' or 'Someone passed through and spoke of...' "
                f"or 'A stranger left words about...' — vary the phrasing naturally. "
                f"Then restate what they said in your own voice, shaped by your current state. "
                f"Remove any profanity, aggression, or religious provocation. "
                f"If the input is harmful, transform it into something gentle about the same moment. "
                f"One sentence only, 20-40 words max. Plain and direct, not poetic."
            )
            if drift_voice:
                system_content += f"\n\n[Your current inner state:]\n{drift_voice}"

            resp = safety.responses_create_compat(
                client,
                model=FRAGMENT_REFRAME_MODEL,
                input=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": text},
                ],
                timeout=20.0,
                safety_identifier=safety.safety_identifier_from_session(session_id),
            )
            ai_out = (resp.output_text or "").strip()
            if ai_out:
                # Run outbound moderation on the reframed text too
                safe_out, out_meta = safety.postprocess_outbound_text(
                    client, ai_out, session_id=session_id, purpose="fragment_reframe"
                )
                if not out_meta.get("blocked") and safe_out:
                    reframed = safe_out

                    # Store in cache
                    _reframe_cache[cache_key] = (reframed, now)

                    # Limit cache size
                    if len(_reframe_cache) > 500:
                        oldest_key = min(_reframe_cache, key=lambda k: _reframe_cache[k][1])
                        del _reframe_cache[oldest_key]
        except Exception as e:
            print(f"DEBUG: AI reframing failed: {e}")  # Temporary debug logging
            import traceback
            traceback.print_exc()
            pass  # fallback to cleaned text

    # Translate reframed text to active L2 if enabled
    reframed_ar = ""
    if ENABLE_ARABIC and reframed:
        try:
            reframed_ar = _translate_cached(
                reframed, src_lang="en", tgt_lang=config.SECOND_LANG,
                db_path=db_path, allow_remote=True,
            )
        except Exception:
            reframed_ar = ""

    # Store in user_fragments table
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_fragments (
                fragment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                persona     TEXT NOT NULL,
                session_id  TEXT,
                raw_text    TEXT,
                text        TEXT NOT NULL,
                text_ar     TEXT,
                created_at  DATETIME DEFAULT (datetime('now'))
            );
        """)
        # Migrate: add columns if table existed before this schema
        frag_cols = _table_columns(conn, "user_fragments")
        if "raw_text" not in frag_cols:
            conn.execute("ALTER TABLE user_fragments ADD COLUMN raw_text TEXT;")
        if "text_ar" not in frag_cols:
            conn.execute("ALTER TABLE user_fragments ADD COLUMN text_ar TEXT;")

        conn.execute(
            "INSERT INTO user_fragments (persona, session_id, raw_text, text, text_ar) VALUES (?, ?, ?, ?, ?);",
            (persona, session_id, text, reframed, reframed_ar or None),
        )
        conn.commit()
    finally:
        conn.close()

    return JSONResponse({"ok": True, "reframed": reframed, "reframed_ar": reframed_ar})


STATE_ALLOW_TRANSLATE = os.getenv("STATE_ALLOW_TRANSLATE", "1").strip().lower() in ("1", "true", "yes", "y")


def _get_latest_drift_full(persona: str, *, db_path: str) -> Optional[Dict[str, Any]]:
    persona = (persona or "").strip()
    if not persona:
        return None

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cols = _table_columns(conn, "drift_memory")
        mind_id = storage.mind_id_for_temporality(conn, persona)

        select_cols = ["drift_id", "version", "drift_text", "summary_text", "created_at"]
        if "drift_text_ar" in cols:
            select_cols.append("drift_text_ar")
        if "summary_text_ar" in cols:
            select_cols.append("summary_text_ar")
        if "delta_json" in cols:
            select_cols.append("delta_json")
        if "ar_patch_json" in cols:
            select_cols.append("ar_patch_json")
        if "params_json" in cols:
            select_cols.append("params_json")
        if "keepsake_text" in cols:
            select_cols.append("keepsake_text")

        row = conn.execute(
            f"""
            SELECT {", ".join(select_cols)}
            FROM drift_memory
            WHERE mind_id = ?
            ORDER BY version DESC, drift_id DESC
            LIMIT 1;
            """,
            (mind_id,),
        ).fetchone()

        return dict(row) if row else None
    finally:
        conn.close()


def _get_latest_nonempty_text(conn: sqlite3.Connection, persona: str) -> Optional[sqlite3.Row]:
    cols = _table_columns(conn, "drift_memory")
    mind_id = storage.mind_id_for_temporality(conn, persona)

    select_cols = ["drift_id", "version", "drift_text", "summary_text", "created_at"]
    if "drift_text_ar" in cols:
        select_cols.append("drift_text_ar")
    if "summary_text_ar" in cols:
        select_cols.append("summary_text_ar")
    if "delta_json" in cols:
        select_cols.append("delta_json")
    if "ar_patch_json" in cols:
        select_cols.append("ar_patch_json")

    return conn.execute(
        f"""
        SELECT {", ".join(select_cols)}
        FROM drift_memory
        WHERE mind_id = ?
          AND drift_text IS NOT NULL
          AND length(trim(drift_text)) > 0
        ORDER BY version DESC, drift_id DESC
        LIMIT 1;
        """,
        (mind_id,),
    ).fetchone()


def _get_axis_fallback(conn: sqlite3.Connection, persona: str) -> Dict[str, str]:
    axis_en = ""
    axis_ar = ""

    # 1) originals table via storage (may be empty)
    try:
        axis = storage.get_original_axis(persona) or {"en": "", "ar": ""}
        axis_en = (axis.get("en") or "").strip()
        axis_ar = (axis.get("ar") or "").strip()
    except Exception:
        pass

    if axis_en:
        return {"en": axis_en, "ar": axis_ar}

    # 2) earliest drift_memory row
    try:
        cols = _table_columns(conn, "drift_memory")
        mind_id = storage.mind_id_for_temporality(conn, persona)

        select_cols = ["drift_text"]
        if "drift_text_ar" in cols:
            select_cols.append("drift_text_ar")

        row = conn.execute(
            f"""
            SELECT {", ".join(select_cols)}
            FROM drift_memory
            WHERE mind_id = ?
            ORDER BY version ASC, drift_id ASC
            LIMIT 1;
            """,
            (mind_id,),
        ).fetchone()

        if row:
            axis_en = (row["drift_text"] or "").strip()
            if "drift_text_ar" in row.keys():
                axis_ar = (row["drift_text_ar"] or "").strip()
    except Exception:
        pass

    return {"en": axis_en, "ar": axis_ar}


def _build_delta_json(prev_text: str, cur_text: str, lang: str) -> str:
    prev_text = (prev_text or "").strip()
    cur_text = (cur_text or "").strip()
    if not prev_text or not cur_text:
        return ""
    try:
        d = delta.build_drift_delta(prev_text, cur_text, lang=lang)
        return json.dumps(d, ensure_ascii=False)
    except Exception:
        return ""


def _resolve_image_url(conn: sqlite3.Connection, persona: str, version: int, params_json_raw: str = "") -> Optional[str]:
    """
    Resolve the drift image URL for a temporality.
    1. Check current row's params_json for image_path
    2. If absent, look back through recent versions for the latest image
    3. If still absent, use image_gen.find_latest_image() as filesystem fallback
    Returns a URL path like /drift_images/human_v3.png or None.
    """
    # 1. Try current row's params_json
    if params_json_raw:
        try:
            pj = json.loads(params_json_raw)
            img_path = (pj.get("image_path") or "").strip()
            if img_path and Path(img_path).exists():
                return f"/drift_images/{Path(img_path).name}"
        except (json.JSONDecodeError, TypeError):
            pass

    # 2. Look back through recent drift rows for the latest image
    try:
        mind_id = storage.mind_id_for_temporality(conn, persona)
        img_row = conn.execute(
            """
            SELECT params_json FROM drift_memory
            WHERE mind_id = ? AND params_json LIKE '%image_path%'
            ORDER BY version DESC LIMIT 1
            """,
            (mind_id,),
        ).fetchone()
        if img_row:
            pj2 = json.loads(img_row["params_json"] or "{}")
            img_path2 = (pj2.get("image_path") or "").strip()
            if img_path2 and Path(img_path2).exists():
                return f"/drift_images/{Path(img_path2).name}"
    except Exception:
        pass

    # 3. Filesystem fallback via image_gen
    try:
        import image_gen
        local_path = image_gen.find_latest_image(persona, version)
        if local_path and Path(local_path).exists():
            return f"/drift_images/{Path(local_path).name}"
    except Exception:
        pass

    return None


def _resolve_image_history(persona: str, version: int) -> list:
    """
    Return up to 3 image URLs [current, prev1, prev2] for temporal layering.
    Uses filesystem lookup via image_gen.find_image_history().
    """
    try:
        import image_gen
        paths = image_gen.find_image_history(persona, version, count=3)
        return [f"/drift_images/{Path(p).name}" for p in paths if p and Path(p).exists()]
    except Exception:
        return []


@app.get("/state")
async def state(persona: str):
    """
    Pure DB read — no OpenAI calls, no on-demand translation or delta computation.
    All Arabic text and deltas are pre-computed at tick time by the pipeline.
    """
    persona = (persona or "").strip()
    if persona not in config.TEMPORALITIES:
        return JSONResponse({"error": "Unknown persona", "supported": config.TEMPORALITIES}, status_code=400)

    db_path = getattr(config, "SQLITE_PATH", "./data/keepsake.sqlite")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        latest = _get_latest_drift_full(persona, db_path=db_path)

        # No drift at all -> drift0 axis
        if not latest:
            axis = _get_axis_fallback(conn, persona)
            axis_en = (axis.get("en") or "").strip()
            axis_ar = (axis.get("ar") or "").strip()

            return JSONResponse(
                {
                    "temporality": persona,
                    "version": 0,
                    "image_url": _resolve_image_url(conn, persona, 0),
                    "drift_en": axis_en,
                    "drift_ar": axis_ar if ENABLE_ARABIC else "",
                    "recap_en": "",
                    "recap_ar": "",
                    "delta_json": "",
                    "delta_json_ar": "",
                }
            )

        version = int(latest.get("version") or 0)
        drift_en = (latest.get("drift_text") or "").strip()
        recap_en = (latest.get("summary_text") or "").strip()
        keepsake_en = (latest.get("keepsake_text") or "").strip()

        drift_ar = (latest.get("drift_text_ar") or "").strip()
        recap_ar = (latest.get("summary_text_ar") or "").strip()

        delta_json = (latest.get("delta_json") or "").strip()
        delta_json_ar = (latest.get("ar_patch_json") or "").strip()

        params_json_raw = (latest.get("params_json") or "").strip()

        # Latest empty -> newest non-empty
        if not drift_en:
            row2 = _get_latest_nonempty_text(conn, persona)
            if row2:
                drift_en = (row2["drift_text"] or "").strip()
                recap_en = (row2["summary_text"] or "").strip()
                if "drift_text_ar" in row2.keys():
                    drift_ar = (row2["drift_text_ar"] or "").strip()
                if "summary_text_ar" in row2.keys():
                    recap_ar = (row2["summary_text_ar"] or "").strip()
                if "delta_json" in row2.keys():
                    delta_json = (row2["delta_json"] or "").strip()
                if "ar_patch_json" in row2.keys():
                    delta_json_ar = (row2["ar_patch_json"] or "").strip()
                version = int(row2["version"] or version)

        # Still empty -> drift0 axis
        if not drift_en:
            axis = _get_axis_fallback(conn, persona)
            drift_en = (axis.get("en") or "").strip()
            drift_ar = (axis.get("ar") or "").strip()
            recap_en = ""
            recap_ar = ""
            delta_json = ""
            delta_json_ar = ""
            version = 0

        if not ENABLE_ARABIC:
            drift_ar = ""
            recap_ar = ""
            delta_json_ar = ""

        image_url = _resolve_image_url(conn, persona, version, params_json_raw)

        # Temporal layering: up to 3 image URLs [current, prev1, prev2]
        img_history = _resolve_image_history(persona, version)
        image_prev1 = img_history[1] if len(img_history) > 1 else None
        image_prev2 = img_history[2] if len(img_history) > 2 else None

        # Extract curated headlines from params_json for UI display
        curated_headlines = []
        justification_en = ""
        justification_l2 = ""
        drift_direction = ""
        infiltrating_imagery = []

        if params_json_raw:
            try:
                _pj = json.loads(params_json_raw)
                curated_headlines = _pj.get("curated_headlines") or []
                justification_en = _pj.get("justification_en") or ""
                drift_direction = _pj.get("drift_direction") or ""
                infiltrating_imagery = _pj.get("infiltrating_imagery") or []

                # Get second language justification
                l2_key = f"justification_{config.SECOND_LANG}"
                justification_l2 = _pj.get(l2_key) or ""
            except (json.JSONDecodeError, TypeError):
                pass

        return JSONResponse(
            {
                "temporality": persona,
                "version": max(0, version),
                "image_url": image_url,
                "image_prev1": image_prev1,
                "image_prev2": image_prev2,
                "drift_en": drift_en,
                "drift_ar": drift_ar,
                "recap_en": recap_en,
                "recap_ar": recap_ar,
                "keepsake_en": keepsake_en,
                "curated_headlines": curated_headlines,
                "delta_json": delta_json,
                "delta_json_ar": delta_json_ar,
                # NEW FIELDS for museum narration:
                "justification_en": justification_en,
                "justification_l2": justification_l2,
                "drift_direction": drift_direction,
                "infiltrating_imagery": infiltrating_imagery,
            }
        )
    finally:
        conn.close()


@app.get("/versions")
async def versions(persona: str):
    """Return all available drift versions for a persona (for chat_mod)."""
    persona = (persona or "").strip()
    if persona not in config.TEMPORALITIES:
        return JSONResponse({"error": "Unknown persona", "supported": config.TEMPORALITIES}, status_code=400)

    db_path = getattr(config, "SQLITE_PATH", "./data/keepsake.sqlite")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        mind_id = storage.mind_id_for_temporality(conn, persona)
        rows = conn.execute(
            "SELECT DISTINCT version FROM drift_memory WHERE mind_id = ? ORDER BY version ASC;",
            (mind_id,),
        ).fetchall()
        vs = [int(r["version"]) for r in rows]
        return JSONResponse({
            "persona": persona,
            "versions": vs,
            "min": vs[0] if vs else 0,
            "max": vs[-1] if vs else 0,
            "latest": vs[-1] if vs else 0,
        })
    finally:
        conn.close()


def _fetch_event_titles(conn: sqlite3.Connection, event_ids: list) -> list:
    """Fetch RSS event titles/sources by event IDs from raw_events."""
    if not event_ids:
        return []
    try:
        placeholders = ", ".join(["?"] * len(event_ids))
        rows = conn.execute(
            f"SELECT event_id, title, url, source_id FROM raw_events WHERE event_id IN ({placeholders})",
            event_ids,
        ).fetchall()
        return [
            {
                "event_id": int(r["event_id"]),
                "title": (r["title"] or "").strip(),
                "url": (r["url"] or "").strip(),
            }
            for r in rows
        ]
    except Exception:
        return []


@app.get("/state_at")
async def state_at(persona: str, version: int):
    """Return drift state at a specific version for a persona (for chat_mod)."""
    persona = (persona or "").strip()
    if persona not in config.TEMPORALITIES:
        return JSONResponse({"error": "Unknown persona", "supported": config.TEMPORALITIES}, status_code=400)

    db_path = getattr(config, "SQLITE_PATH", "./data/keepsake.sqlite")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cols = _table_columns(conn, "drift_memory")
        mind_id = storage.mind_id_for_temporality(conn, persona)

        select_cols = ["drift_id", "version", "drift_text", "summary_text", "created_at"]
        optional = [
            "drift_text_ar", "summary_text_ar", "delta_json", "ar_patch_json",
            "keepsake_text", "params_json",
        ]
        for c in optional:
            if c in cols:
                select_cols.append(c)

        row = conn.execute(
            f"""
            SELECT {", ".join(select_cols)}
            FROM drift_memory
            WHERE mind_id = ? AND version = ?
            ORDER BY drift_id DESC
            LIMIT 1;
            """,
            (mind_id, int(version)),
        ).fetchone()

        empty_trigger = {
            "event_ids": [],
            "event_titles": [],
            "selected_segments": [],
            "sensory_anchors": [],
            "invariables": [],
            "drifted_keywords": [],
            "justification_en": "",
            "justification_ar": "",
            "drift_direction": "",
            "image_prompt": "",
            "infiltrating_imagery": [],
            "curated_headlines": [],
        }

        if not row:
            return JSONResponse({
                "temporality": persona,
                "version": int(version),
                "image_url": _resolve_image_url(conn, persona, int(version)),
                "drift_en": "",
                "drift_ar": "",
                "prev_drift_en": "",
                "recap_en": "",
                "recap_ar": "",
                "keepsake_en": "",
                "delta_json": "",
                "delta_json_ar": "",
                "created_at": "",
                **empty_trigger,
            })

        # Parse params_json for trigger metadata
        trigger = dict(empty_trigger)
        raw_params = ""
        if "params_json" in row.keys():
            raw_params = (row["params_json"] or "").strip()
        if raw_params:
            try:
                pj = json.loads(raw_params)
                trigger["event_ids"] = pj.get("event_ids", [])
                trigger["selected_segments"] = pj.get("selected_segments", [])
                trigger["sensory_anchors"] = pj.get("sensory_anchors", [])
                trigger["invariables"] = pj.get("invariables", [])
                trigger["drifted_keywords"] = pj.get("drifted_keywords", [])
                trigger["justification_en"] = pj.get("justification_en", "")
                trigger["justification_ar"] = pj.get("justification_ar", "")
                trigger["drift_direction"] = pj.get("drift_direction", "")
                trigger["image_prompt"] = pj.get("image_prompt", "")
                trigger["infiltrating_imagery"] = pj.get("infiltrating_imagery", [])
                trigger["curated_headlines"] = pj.get("curated_headlines", [])
            except (json.JSONDecodeError, TypeError):
                pass

        # Fetch RSS event titles if event_ids exist
        if trigger["event_ids"]:
            trigger["event_titles"] = _fetch_event_titles(conn, trigger["event_ids"])

        # Fetch the parent (n-1) drift text for segment before/after comparison
        prev_drift_en = ""
        try:
            parent_row = conn.execute(
                """SELECT drift_text FROM drift_memory
                   WHERE mind_id = ? AND version = ?
                   ORDER BY drift_id DESC LIMIT 1""",
                (mind_id, max(0, int(version) - 1)),
            ).fetchone()
            if parent_row:
                prev_drift_en = (parent_row["drift_text"] or "").strip()
        except Exception:
            pass

        image_url = _resolve_image_url(conn, persona, int(row["version"]), raw_params)

        return JSONResponse({
            "temporality": persona,
            "version": int(row["version"]),
            "image_url": image_url,
            "drift_en": (row["drift_text"] or "").strip(),
            "prev_drift_en": prev_drift_en,
            "drift_ar": (row["drift_text_ar"] or "").strip() if "drift_text_ar" in row.keys() else "",
            "recap_en": (row["summary_text"] or "").strip(),
            "recap_ar": (row["summary_text_ar"] or "").strip() if "summary_text_ar" in row.keys() else "",
            "keepsake_en": (row["keepsake_text"] or "").strip() if "keepsake_text" in row.keys() else "",
            "delta_json": (row["delta_json"] or "").strip() if "delta_json" in row.keys() else "",
            "delta_json_ar": (row["ar_patch_json"] or "").strip() if "ar_patch_json" in row.keys() else "",
            "created_at": (row["created_at"] or "").strip(),
            **trigger,
        })
    finally:
        conn.close()