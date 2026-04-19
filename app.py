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

try:
    import llm_logger
except Exception:
    llm_logger = None  # type: ignore

# ── Pillow auto-install ────────────────────────────────────────────────────────
# Pillow is required for JPEG conversion in /drift_images/. If it's missing from
# the venv (e.g. added to requirements.txt after the venv was created), install
# it automatically at startup so images start serving as JPEG immediately.
try:
    from PIL import Image as _PILCheck  # noqa: F401  — just a presence check
except ImportError:
    import subprocess as _subp, sys as _sys
    _log_startup = logging.getLogger("startup")
    _log_startup.info("Pillow not found — installing via pip …")
    _r = _subp.run(
        [_sys.executable, "-m", "pip", "install", "Pillow", "--quiet"],
        capture_output=True, text=True,
    )
    if _r.returncode == 0:
        _log_startup.info("Pillow installed OK")
    else:
        _log_startup.warning("Pillow pip install failed: %s", _r.stderr[:200])

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
    allow_origin_regex=r"https://([a-zA-Z0-9-]+\.)?(keepsake-drift|keepsake-br|keepsake-gr|keepsake-czi)\.pages\.dev|https://([a-zA-Z0-9-]+\.)?keepsake-drift\.net",
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods including OPTIONS
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

# CORS middleware handles OPTIONS preflight automatically — no catch-all handler needed

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Serve CSS/JS/images at root paths too (HTML uses relative paths like "css/base.css")
# This makes the tunnel URL serve the frontend correctly alongside the API
for _subdir in ("css", "js", "images"):
    _subpath = STATIC_DIR / _subdir
    if _subpath.is_dir():
        app.mount(f"/{_subdir}", StaticFiles(directory=str(_subpath)), name=f"root-{_subdir}")

# Serve generated drift images from data/images/ (with no-cache headers)
DATA_IMAGES_DIR = BASE_DIR / "data" / "images"
DATA_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/drift_images/{filename}")
async def serve_drift_image(filename: str, mobile: str = ""):
    """
    Serve drift images as compressed JPEG for all clients.
    - Desktop: max 1200px wide, quality 85  (~300–500 KB vs 2–3 MB PNG)
    - Mobile (mobile=1): max 700px wide, quality 75  (~80–150 KB)
    Images are versioned in the URL (human_v65.png) so they're immutable —
    cached at Cloudflare edge for 7 days.
    """
    img_path = DATA_IMAGES_DIR / filename
    if not img_path.exists() or not img_path.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)

    # Versioned images are immutable — cache aggressively at Cloudflare edge.
    # The ?t= query param in the URL changes whenever a new version is generated,
    # so stale content is never served.
    cache_headers = {
        "Cache-Control": "public, max-age=604800, stale-while-revalidate=86400",
    }

    is_mobile = mobile == "1"
    max_width = 700 if is_mobile else 1200
    quality   = 75  if is_mobile else 85

    jpeg_dir  = DATA_IMAGES_DIR / ("mobile" if is_mobile else "jpeg")
    jpeg_dir.mkdir(exist_ok=True)
    stem      = Path(filename).stem
    jpeg_path = jpeg_dir / f"{stem}.jpg"

    # Regenerate JPEG if missing or PNG is newer (e.g. image was just regenerated)
    needs_regen = (
        not jpeg_path.exists()
        or jpeg_path.stat().st_mtime < img_path.stat().st_mtime
    )
    if needs_regen:
        converted = False
        # Try Pillow first
        try:
            from PIL import Image as PILImage
            with PILImage.open(img_path) as im:
                w, h = im.size
                if w > max_width:
                    new_h = int(h * max_width / w)
                    im = im.resize((max_width, new_h), PILImage.LANCZOS)
                im = im.convert("RGB")
                im.save(jpeg_path, "JPEG", quality=quality, optimize=True)
            converted = True
        except Exception as pil_err:
            _tick_log.warning("PIL conversion failed (%s), trying ImageMagick", pil_err)

        # Fallback: ImageMagick convert (pre-installed on Raspberry Pi OS)
        if not converted:
            try:
                import subprocess
                resize_arg = f"{max_width}x>"  # only shrink, never enlarge
                result = subprocess.run(
                    [
                        "convert", str(img_path),
                        "-resize", resize_arg,
                        "-quality", str(quality),
                        str(jpeg_path),
                    ],
                    capture_output=True, timeout=30,
                )
                if result.returncode == 0 and jpeg_path.exists():
                    converted = True
                else:
                    _tick_log.warning(
                        "ImageMagick convert failed: %s", result.stderr.decode()
                    )
            except Exception as im_err:
                _tick_log.warning("ImageMagick unavailable: %s", im_err)

        if not converted:
            # Both methods failed — serve original PNG (no cache break, file unchanged)
            return FileResponse(img_path, headers=cache_headers)

    return FileResponse(jpeg_path, media_type="image/jpeg", headers=cache_headers)

_client: Optional[OpenAI] = None


_tick_log = logging.getLogger("tick_loop")

# ── Global tick cycle ──────────────────────────────────────────────
# TICK_DISPLAY_INTERVAL: how often the frontend shows a new drift (seconds)
# TICK_LEAD_TIME: how many seconds BEFORE display to start processing
#   (ingest RSS + run drift pipeline).  Processing takes ~3 min,
#   so a 5-min lead gives comfortable headroom.
TICK_DISPLAY_INTERVAL = int(os.getenv("KD_TICK_INTERVAL", "1800"))   # 30 min
TICK_LEAD_TIME = int(os.getenv("KD_TICK_LEAD", "300"))              # 5 min before
TICK_ENABLED = os.getenv("KD_TICK_ENABLED", "0").strip().lower() in ("1", "true", "yes")

# ── Continuous drift loop ────────────────────────────────────────
DRIFT_CYCLE_SECONDS = int(os.getenv("KD_DRIFT_CYCLE", "300"))         # 5 min
DRIFT_MIN_INTERVAL = int(os.getenv("KD_DRIFT_MIN_INTERVAL", "600"))   # 10 min per-mind
IMAGE_DRIFT_THRESHOLD = float(os.getenv("KD_IMAGE_DRIFT_THRESHOLD", "0.4"))
CONTINUOUS_DRIFT_ENABLED = os.getenv("KD_CONTINUOUS_ENABLED", "1").strip().lower() in ("1", "true", "yes")

_tick_thread: Optional[threading.Thread] = None
_tick_stop = threading.Event()

# ── Auto-drift toggle (persistent flag file) ─────────────────────
from pathlib import Path as _Path
_AUTO_DRIFT_FLAG = _Path(os.path.dirname(os.path.abspath(__file__))) / "data" / "auto_drift.flag"

def _auto_drift_enabled() -> bool:
    return _AUTO_DRIFT_FLAG.exists()

def _set_auto_drift(enabled: bool) -> None:
    if enabled:
        _AUTO_DRIFT_FLAG.parent.mkdir(parents=True, exist_ok=True)
        _AUTO_DRIFT_FLAG.touch()
    else:
        _AUTO_DRIFT_FLAG.unlink(missing_ok=True)

# Default ON at first launch (flag doesn't exist yet)
if not _AUTO_DRIFT_FLAG.parent.exists():
    _AUTO_DRIFT_FLAG.parent.mkdir(parents=True, exist_ok=True)
if not _AUTO_DRIFT_FLAG.exists():
    try:
        _AUTO_DRIFT_FLAG.touch()
    except Exception:
        pass

# ── Image queue (text drift doesn't block on image gen) ──────────
import queue as _queue
_image_queue: "_queue.Queue[tuple]" = _queue.Queue()
_image_worker_thread: Optional[threading.Thread] = None

# Last pick tracking for admin status
_last_drift_pick: Dict[str, Any] = {"mind": None, "at": None}

# ── Tunnel watchdog (independent of tick loop) ────────────────────
_tunnel_thread: Optional[threading.Thread] = None
_tunnel_stop = threading.Event()
TUNNEL_CHECK_INTERVAL = int(os.getenv("KD_TUNNEL_CHECK_INTERVAL", "30"))  # seconds


def _run_one_tick() -> None:
    """Execute one full cycle: OpenClaw prepare → drift → save + image gen."""
    try:
        import tick_openclaw
        import importlib
        importlib.reload(tick_openclaw)

        # Phase 1: Prepare (RSS ingest + realtime APIs + DB state → context JSON)
        _tick_log.info("Tick loop: OpenClaw prepare …")

        class _Args:
            pass
        args = _Args()

        tick_openclaw.cmd_prepare(args)
        _tick_log.info("Tick loop: prepare done.")

        # Phase 2: Drift (bundled OpenAI call, all 6 minds, temperature=0.9)
        _tick_log.info("Tick loop: OpenClaw drift …")
        tick_openclaw.cmd_drift(args)
        _tick_log.info("Tick loop: drift done.")

        # Phase 3: Save (validate, compute deltas, write to DB)
        _tick_log.info("Tick loop: OpenClaw save …")
        tick_openclaw.cmd_save(args)
        _tick_log.info("Tick loop: save done.")

        # Phase 4: Image generation (uses the new drift text + scene blending)
        _tick_log.info("Tick loop: generating images …")
        try:
            db_path = getattr(config, "SQLITE_PATH", "./data/keepsake.sqlite")
            import sqlite3 as _sql
            import json as _json
            _conn = _sql.connect(db_path)
            _conn.row_factory = _sql.Row
            _minds = _conn.execute("SELECT mind_id, mind_key FROM minds").fetchall()
            import image_gen

            # Collect per-mind imagery, drift text, resonance for scene blending
            _latest_per_mind = {}
            _imagery_per_mind = {}
            _drift_texts_per_mind = {}
            _resonance_per_mind = {}
            for mr in _minds:
                _latest = _conn.execute(
                    "SELECT version, drift_text, params_json FROM drift_memory WHERE mind_id=? ORDER BY version DESC LIMIT 1",
                    (mr["mind_id"],),
                ).fetchone()
                if not _latest:
                    continue
                _pj = _json.loads(_latest["params_json"] or "{}")
                _latest_per_mind[mr["mind_key"]] = (mr["mind_id"], _latest, _pj)
                _imagery_per_mind[mr["mind_key"]] = _pj.get("infiltrating_imagery", [])
                _drift_texts_per_mind[mr["mind_key"]] = _latest["drift_text"]
                _resonance_per_mind[mr["mind_key"]] = float(_pj.get("resonance", 0.5))

            # Blend scenes: transform each mind's base scene with its infiltrating imagery
            # This produces scene/weather/time_of_day/color_palette per mind that vary by tick
            _blend_result = {}
            try:
                _blend_result = image_gen.blend_scenes_batch(
                    infiltrating_per_mind=_imagery_per_mind,
                    drift_texts=_drift_texts_per_mind,
                    resonance_per_mind=_resonance_per_mind,
                )
                _tick_log.info("Tick loop: blended scenes for %d minds", len(_blend_result))
            except Exception:
                _tick_log.exception("Tick loop: scene blending failed (non-fatal)")

            for mind_key, (mind_id, _latest, _pj) in _latest_per_mind.items():
                if not image_gen.should_generate(_latest["version"]):
                    continue
                _blend = _blend_result.get(mind_key, {}) if _blend_result else {}
                path, _ = image_gen.generate_drift_image(
                    mind_key=mind_key,
                    version=_latest["version"],
                    drift_text=_latest["drift_text"],
                    drift_direction=_pj.get("drift_direction", ""),
                    perspective=_pj.get("perspective", ""),
                    infiltrating_imagery=_pj.get("infiltrating_imagery", []),
                    blended_scene=_blend.get("scene") if _blend else None,
                    visual_atmosphere=_blend if _blend else None,
                )
                if path:
                    _tick_log.info("Image: %s v%d OK", mind_key, _latest["version"])
                    _pj["image_path"] = str(path)
                    _conn.execute(
                        "UPDATE drift_memory SET params_json=? WHERE mind_id=? AND version=?",
                        (_json.dumps(_pj, ensure_ascii=False), mind_id, _latest["version"]),
                    )
                    _conn.commit()
                else:
                    _tick_log.warning("Image: %s v%d FAILED", mind_key, _latest["version"])
            _conn.close()
        except Exception:
            _tick_log.exception("Tick loop: image generation error (non-fatal)")

        _tick_log.info("Tick loop: complete.")

    except Exception:
        _tick_log.exception("Tick loop: error during tick execution")

    # Phase 3: Tunnel health check (runs even if drift failed)
    try:
        import tunnel_watchdog
        wd = tunnel_watchdog.check_and_heal()
        if wd["action"] != "none":
            _tick_log.info("Tunnel watchdog: action=%s url=%s", wd["action"], wd["url"])
    except Exception:
        _tick_log.debug("Tunnel watchdog: not available or failed")


def _pick_mind_to_drift() -> Optional[str]:
    """Choose which mind should drift this cycle.

    Priority:
    1. Visitor fragment added since last drift for that mind → that mind.
    2. Round-robin by staleness — oldest mind that hasn't drifted in
       DRIFT_MIN_INTERVAL seconds.
    3. None if every mind drifted too recently.
    """
    try:
        import tick_openclaw
        # Check visitor fragments: any fragment newer than the mind's last drift?
        import sqlite3 as _sql
        db_path = getattr(config, "SQLITE_PATH", "./data/keepsake.sqlite")
        conn = _sql.connect(db_path)
        conn.row_factory = _sql.Row
        try:
            # For each mind, last drift time and last visitor fragment time
            rows = conn.execute("""
                SELECT m.mind_key,
                       (SELECT MAX(created_at) FROM drift_memory WHERE mind_id=m.mind_id) AS last_drift,
                       (SELECT MAX(created_at) FROM user_fragments WHERE persona=m.mind_key) AS last_visitor
                  FROM minds m
            """).fetchall()
            for r in rows:
                lv = r["last_visitor"]
                ld = r["last_drift"]
                if lv and (not ld or lv > ld):
                    return r["mind_key"]
        finally:
            conn.close()

        # Round-robin staleness
        stale = tick_openclaw.list_minds_by_staleness()
        for mkey, age in stale:
            if age >= DRIFT_MIN_INTERVAL:
                return mkey
        return None
    except Exception:
        _tick_log.exception("_pick_mind_to_drift failed")
        return None


def _cumulative_drift_distance(mind_key: str) -> float:
    """Sum of drift_distance across rows since the last row with image_path set."""
    import sqlite3 as _sql
    import json as _json
    db_path = getattr(config, "SQLITE_PATH", "./data/keepsake.sqlite")
    conn = _sql.connect(db_path)
    conn.row_factory = _sql.Row
    try:
        mr = conn.execute("SELECT mind_id FROM minds WHERE mind_key=?", (mind_key,)).fetchone()
        if not mr:
            return 0.0
        rows = conn.execute(
            "SELECT params_json FROM drift_memory WHERE mind_id=? ORDER BY version DESC LIMIT 40",
            (mr["mind_id"],),
        ).fetchall()
        total = 0.0
        for r in rows:
            try:
                pj = _json.loads(r["params_json"] or "{}")
            except Exception:
                pj = {}
            # Stop once we hit a row that has image_path (=last image generated here)
            if pj.get("image_path"):
                break
            d = pj.get("drift_distance")
            if isinstance(d, (int, float)):
                total += float(d)
        return total
    finally:
        conn.close()


def _run_one_mind_drift(mkey: str) -> None:
    """Drift one mind, then conditionally queue an image regen."""
    import tick_openclaw
    _last_drift_pick["mind"] = mkey
    _last_drift_pick["at"] = time.time()
    _tick_log.info("continuous: drifting %s", mkey)
    new_version = tick_openclaw.drift_one_mind(mkey)
    if new_version is None:
        return
    try:
        cumulative = _cumulative_drift_distance(mkey)
        _tick_log.info("continuous: %s v%d, cumulative distance=%.3f (threshold %.2f)",
                       mkey, new_version, cumulative, IMAGE_DRIFT_THRESHOLD)
        if cumulative >= IMAGE_DRIFT_THRESHOLD:
            _image_queue.put((mkey, new_version))
            _tick_log.info("continuous: queued image regen for %s v%d", mkey, new_version)
    except Exception:
        _tick_log.exception("continuous: post-drift image check failed for %s", mkey)


def _image_worker_loop() -> None:
    """Background worker that consumes image gen requests from _image_queue."""
    import json as _json
    import sqlite3 as _sql
    _tick_log.info("image worker started")
    while not _tick_stop.is_set():
        try:
            item = _image_queue.get(timeout=5)
        except Exception:
            continue
        if item is None:
            break
        mkey, version = item
        try:
            db_path = getattr(config, "SQLITE_PATH", "./data/keepsake.sqlite")
            conn = _sql.connect(db_path)
            conn.row_factory = _sql.Row
            mr = conn.execute("SELECT mind_id FROM minds WHERE mind_key=?", (mkey,)).fetchone()
            if not mr:
                conn.close()
                continue
            row = conn.execute(
                "SELECT drift_text, params_json FROM drift_memory WHERE mind_id=? AND version=?",
                (mr["mind_id"], version),
            ).fetchone()
            if not row:
                conn.close()
                continue
            pj = _json.loads(row["params_json"] or "{}")
            import image_gen
            blend_result = {}
            try:
                blend_result = image_gen.blend_scenes_batch(
                    infiltrating_per_mind={mkey: pj.get("infiltrating_imagery", [])},
                    drift_texts={mkey: row["drift_text"]},
                    resonance_per_mind={mkey: float(pj.get("resonance", 0.5))},
                )
            except Exception:
                _tick_log.exception("image worker: blend failed for %s v%d", mkey, version)
            b = blend_result.get(mkey, {}) if blend_result else {}
            path, _ = image_gen.generate_drift_image(
                mind_key=mkey, version=version, drift_text=row["drift_text"],
                drift_direction=pj.get("drift_direction", ""),
                perspective=pj.get("perspective", ""),
                infiltrating_imagery=pj.get("infiltrating_imagery", []),
                blended_scene=b.get("scene") if b else None,
                visual_atmosphere=b if b else None,
            )
            if path:
                pj["image_path"] = str(path)
                pj["image_version"] = version
                conn.execute(
                    "UPDATE drift_memory SET params_json=? WHERE mind_id=? AND version=?",
                    (_json.dumps(pj, ensure_ascii=False), mr["mind_id"], version),
                )
                conn.commit()
                _tick_log.info("image worker: %s v%d OK", mkey, version)
            else:
                _tick_log.warning("image worker: %s v%d FAILED", mkey, version)
            conn.close()
        except Exception:
            _tick_log.exception("image worker: error processing %s v%s", mkey, version)
        finally:
            try:
                _image_queue.task_done()
            except Exception:
                pass


def _continuous_drift_loop() -> None:
    """Continuous drift: every DRIFT_CYCLE_SECONDS pick at most one mind."""
    _tick_log.info("continuous drift loop started (cycle=%ds, min_interval=%ds)",
                   DRIFT_CYCLE_SECONDS, DRIFT_MIN_INTERVAL)
    while not _tick_stop.is_set():
        try:
            if _auto_drift_enabled():
                mkey = _pick_mind_to_drift()
                if mkey:
                    _run_one_mind_drift(mkey)
                else:
                    _tick_log.debug("continuous: idle cycle (no mind due)")
            else:
                _tick_log.debug("continuous: auto-drift disabled")
        except Exception:
            _tick_log.exception("continuous drift cycle failed")
        _tick_stop.wait(DRIFT_CYCLE_SECONDS)


def _tick_loop() -> None:
    """Legacy wrapper — delegates to continuous drift loop."""
    _continuous_drift_loop()


def _tunnel_watchdog_loop() -> None:
    """
    Background thread that keeps the NAMED Cloudflare tunnel alive.
    Checks every TUNNEL_CHECK_INTERVAL seconds if the cloudflared process
    for the named tunnel is still running, and restarts it if not.
    """
    import subprocess as _sp

    # Detect named tunnel config from ~/.cloudflared/
    _port = int(os.getenv("PORT", "8000"))
    _tunnel_configs = {
        8000: ("drift-ar", os.path.expanduser("~/.cloudflared/config-drift-ar.yml")),
        8001: ("drift-en", os.path.expanduser("~/.cloudflared/config-drift-en.yml")),
        8002: ("drift-gr", os.path.expanduser("~/.cloudflared/config-drift-gr.yml")),
        8003: ("drift-br", os.path.expanduser("~/.cloudflared/config-drift-br.yml")),
    }
    _tunnel_name, _tunnel_cfg = _tunnel_configs.get(_port, (None, None))

    if not _tunnel_name or not _tunnel_cfg or not os.path.exists(_tunnel_cfg):
        _tick_log.info("No named tunnel config for port %d — falling back to quick tunnel watchdog.", _port)
        # Fall back to old quick tunnel watchdog
        _tunnel_stop.wait(timeout=10)
        while not _tunnel_stop.is_set():
            try:
                import tunnel_watchdog
                wd = tunnel_watchdog.check_and_heal()
                if wd["action"] != "none":
                    _tick_log.info("Tunnel watchdog (bg): action=%s url=%s", wd["action"], wd["url"])
            except Exception as e:
                _tick_log.debug("Tunnel watchdog (bg): error — %s", e)
            _tunnel_stop.wait(timeout=TUNNEL_CHECK_INTERVAL)
        return

    _tick_log.info(
        "Named tunnel watchdog started — monitoring '%s' every %ds.",
        _tunnel_name, TUNNEL_CHECK_INTERVAL,
    )
    _tunnel_stop.wait(timeout=15)  # let server fully start

    _cloudflared_bin = "/opt/homebrew/bin/cloudflared"
    if not os.path.exists(_cloudflared_bin):
        _cloudflared_bin = "cloudflared"

    while not _tunnel_stop.is_set():
        try:
            # Check if named tunnel process is running
            result = _sp.run(
                ["pgrep", "-f", f"cloudflared.*{_tunnel_name}"],
                capture_output=True, text=True, timeout=5,
            )
            alive = bool(result.stdout.strip())

            if not alive:
                _tick_log.warning("Named tunnel '%s' is DEAD — restarting...", _tunnel_name)
                _sp.Popen(
                    [_cloudflared_bin, "tunnel", "--config", _tunnel_cfg, "run", _tunnel_name],
                    stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
                )
                _tick_log.info("Restarted named tunnel '%s'.", _tunnel_name)
        except Exception as e:
            _tick_log.debug("Named tunnel watchdog error: %s", e)

        _tunnel_stop.wait(timeout=TUNNEL_CHECK_INTERVAL)


@app.on_event("startup")
def _startup():
    storage.init_db()

    # Launch background continuous-drift loop (always on — internal toggle via flag file)
    global _tick_thread
    if CONTINUOUS_DRIFT_ENABLED and _tick_thread is None:
        _tick_thread = threading.Thread(target=_continuous_drift_loop, daemon=True, name="continuous-drift")
        _tick_thread.start()
        _tick_log.info("Continuous drift loop launched (cycle=%ds).", DRIFT_CYCLE_SECONDS)

    # Launch image worker
    global _image_worker_thread
    if _image_worker_thread is None:
        _image_worker_thread = threading.Thread(target=_image_worker_loop, daemon=True, name="image-worker")
        _image_worker_thread.start()
        _tick_log.info("Image worker thread launched.")

    # Launch tunnel watchdog thread — skip if KD_TUNNEL_WATCHDOG=0
    # (set to 0 when an external service like keepsake-tunnel.service manages the tunnel)
    global _tunnel_thread
    if os.getenv("KD_TUNNEL_WATCHDOG", "1") != "0" and _tunnel_thread is None:
        _tunnel_thread = threading.Thread(
            target=_tunnel_watchdog_loop, daemon=True, name="tunnel-watchdog",
        )
        _tunnel_thread.start()
        _tick_log.info("Tunnel watchdog thread launched (every %ds).", TUNNEL_CHECK_INTERVAL)
    elif os.getenv("KD_TUNNEL_WATCHDOG", "1") == "0":
        _tick_log.info("Tunnel watchdog disabled (KD_TUNNEL_WATCHDOG=0) — external service manages tunnel.")


@app.on_event("shutdown")
def _shutdown():
    _tick_stop.set()
    _tunnel_stop.set()
    if _tick_thread is not None:
        _tick_thread.join(timeout=5)
    if _tunnel_thread is not None:
        _tunnel_thread.join(timeout=5)


def _serve_html(path: Path) -> FileResponse:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Missing HTML file: {path}")
    return FileResponse(
        str(path),
        media_type="text/html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


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
    import time as _time

    src_text = (src_text or "").strip()
    if not src_text:
        return ""
    if src_lang == tgt_lang:
        return src_text

    cached = storage.get_cached_translation(src_lang, tgt_lang, src_text, db_path=db_path)
    if cached:
        if llm_logger:
            llm_logger.log_call(operation="app_translate", backend="cache", model="", latency_ms=0, success=True)
        return cached

    if not allow_remote:
        return ""

    # --- Try OpenAI ---
    client = _client_or_none()
    t0 = _time.time()

    if client is not None:
        prompt = (
            f"Translate the following text from {src_lang} to {tgt_lang}. "
            f"Return only the translation. Preserve paragraph breaks.\n\n{src_text}"
        )
        try:
            openai_timeout = min(30.0, float(config.OPENAI_TIMEOUT_FAST)) if config.OLLAMA_ENABLED else 30.0
            resp = client.responses.create(
                model=TRANSLATE_MODEL,
                input=prompt,
                timeout=openai_timeout,
            )
            out = (resp.output_text or "").strip()
            latency = int((_time.time() - t0) * 1000)
            if llm_logger:
                llm_logger.log_call(operation="app_translate", backend="openai", model=TRANSLATE_MODEL, latency_ms=latency, success=True)
            if out:
                storage.put_cached_translation(src_lang, tgt_lang, src_text, out, db_path=db_path)
            return out
        except Exception as openai_err:
            latency = int((_time.time() - t0) * 1000)
            if llm_logger:
                llm_logger.log_call(
                    operation="app_translate", backend="openai", model=TRANSLATE_MODEL,
                    latency_ms=latency, success=False,
                    error_msg=f"{type(openai_err).__name__}: {str(openai_err)[:200]}",
                )

    # --- Try Ollama fallback ---
    if config.OLLAMA_ENABLED:
        import json, urllib.request
        t1 = _time.time()
        try:
            base_url = config.OLLAMA_BASE_URL.rstrip("/")
            ollama_model = config.OLLAMA_TRANSLATE_MODEL
            payload = json.dumps({
                "model": ollama_model,
                "messages": [
                    {"role": "system", "content": "You are a translator. Return only the translated text."},
                    {"role": "user", "content": f"Translate from {src_lang} to {tgt_lang}. Return only the translation.\n\n{src_text}"},
                ],
                "stream": False,
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{base_url}/api/chat", data=payload,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=30.0) as resp_ol:
                body = json.loads(resp_ol.read().decode("utf-8"))
            out = (body.get("message", {}).get("content") or "").strip()
            latency2 = int((_time.time() - t1) * 1000)
            if llm_logger:
                llm_logger.log_call(
                    operation="app_translate", backend="ollama", model=ollama_model,
                    latency_ms=latency2, success=True,
                )
            if out:
                storage.put_cached_translation(src_lang, tgt_lang, src_text, out, db_path=db_path)
            return out
        except Exception as ollama_err:
            latency2 = int((_time.time() - t1) * 1000)
            if llm_logger:
                llm_logger.log_call(
                    operation="app_translate", backend="ollama", model=config.OLLAMA_TRANSLATE_MODEL,
                    latency_ms=latency2, success=False,
                    error_msg=f"{type(ollama_err).__name__}: {str(ollama_err)[:200]}",
                )

    return ""


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
MUSEUM_DIGITAL_HTML = STATIC_DIR / "museum_digital.html"
MUSEUM_INFRA_HTML = STATIC_DIR / "museum_infrastructure.html"
MUSEUM_MTH_HTML = STATIC_DIR / "museum_more_than_human.html"
MUSEUM_DISPLAY_HTML = STATIC_DIR / "museum_display.html"
MUSEUM1_HTML = STATIC_DIR / "museum1.html"
MUSEUM2_HTML = STATIC_DIR / "museum2.html"
MUSEUM3_HTML = STATIC_DIR / "museum3.html"
PROJECTION_HTML = STATIC_DIR / "projection.html"

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

@app.get("/museum/digital")
async def museum_digital():
    if not MUSEUM_DIGITAL_HTML.exists():
        raise HTTPException(status_code=404, detail="Museum digital stream not found")
    return FileResponse(MUSEUM_DIGITAL_HTML)

@app.get("/museum/infrastructure")
async def museum_infrastructure():
    if not MUSEUM_INFRA_HTML.exists():
        raise HTTPException(status_code=404, detail="Museum infrastructure stream not found")
    return FileResponse(MUSEUM_INFRA_HTML)

@app.get("/museum/more_than_human")
async def museum_more_than_human():
    if not MUSEUM_MTH_HTML.exists():
        raise HTTPException(status_code=404, detail="Museum more-than-human stream not found")
    return FileResponse(MUSEUM_MTH_HTML)

# Dual-temporality display — cycles between two temporalities
# Usage: /museum/display?t1=human&t2=liminal
@app.get("/museum/display")
async def museum_display():
    if not MUSEUM_DISPLAY_HTML.exists():
        raise HTTPException(status_code=404, detail="Museum display not found")
    return FileResponse(MUSEUM_DISPLAY_HTML)

# Named museum displays — 3 screens covering all 6 temporalities
# museum1: human + liminal | museum2: environment + digital | museum3: infrastructure + more_than_human
@app.get("/museum1")
async def museum1():
    if not MUSEUM1_HTML.exists():
        raise HTTPException(status_code=404, detail="Museum 1 page not found")
    return FileResponse(MUSEUM1_HTML)

@app.get("/museum2")
async def museum2():
    if not MUSEUM2_HTML.exists():
        raise HTTPException(status_code=404, detail="Museum 2 page not found")
    return FileResponse(MUSEUM2_HTML)

@app.get("/museum3")
async def museum3():
    if not MUSEUM3_HTML.exists():
        raise HTTPException(status_code=404, detail="Museum 3 page not found")
    return FileResponse(MUSEUM3_HTML)

# Projection — EN+AR drift texts cycling all 6 temporalities
@app.get("/projection")
async def projection():
    if not PROJECTION_HTML.exists():
        raise HTTPException(status_code=404, detail="Projection page not found")
    return FileResponse(PROJECTION_HTML)


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
                    "curated_headlines_ar": [],
                    "realtime_interpretations": [],
                }
                if has_params:
                    raw_pj = (r["params_json"] or "").strip()
                    if raw_pj:
                        try:
                            pj = json.loads(raw_pj)
                            entry["curated_headlines"] = pj.get("curated_headlines") or []
                            entry["curated_headlines_ar"] = pj.get("curated_headlines_ar") or []
                            entry["realtime_interpretations"] = pj.get("realtime_interpretations") or []
                            entry["realtime_source_count"] = len(pj.get("realtime_interpretations") or [])
                            entry["keepsake_text_ar"] = (pj.get("keepsake_text_ar") or "").strip()
                            entry["resonance"] = float(pj.get("resonance", 0.5))
                        except (json.JSONDecodeError, TypeError):
                            pass
                entries.append(entry)

        # 2) Visitor fragment entries
        frag_cols = _table_columns(conn, "user_fragments")
        if frag_cols:  # table exists
            has_text_ar = "text_ar" in frag_cols
            has_tagged = "tagged_text" in frag_cols
            has_tagged_ar = "tagged_text_ar" in frag_cols
            has_resonance = "resonance" in frag_cols
            select_parts = ["text", "created_at"]
            if has_text_ar:
                select_parts.append("text_ar")
            if has_tagged:
                select_parts.append("tagged_text")
            if has_tagged_ar:
                select_parts.append("tagged_text_ar")
            if has_resonance:
                select_parts.append("resonance")
            frag_select = ", ".join(select_parts)

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
                entry = {
                    "source": "visitor",
                    "version": None,
                    "keepsake_text": (fr["text"] or "").strip(),
                    "keepsake_text_ar": (fr["text_ar"] or "").strip() if has_text_ar else "",
                    "created_at": (fr["created_at"] or "").strip(),
                    "curated_headlines": [],
                    "curated_headlines_ar": [],
                }
                if has_tagged and fr["tagged_text"]:
                    entry["tagged_text"] = (fr["tagged_text"] or "").strip()
                if has_tagged_ar and fr["tagged_text_ar"]:
                    entry["tagged_text_ar"] = (fr["tagged_text_ar"] or "").strip()
                if has_resonance and fr["resonance"] is not None:
                    entry["resonance"] = float(fr["resonance"])
                entries.append(entry)

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
    tagged_text = (payload.get("tagged_text") or "").strip()[:120]  # drift fragment they tapped
    tagged_lang = (payload.get("tagged_lang") or "en").strip()[:10]

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
    fragment_resonance = 0.5  # default resonance

    # Check reframe cache (1-hour TTL)
    cache_key = hashlib.sha256(f"{persona}:{text}:{tagged_text}:{drift_voice[:200]}".encode()).hexdigest()
    now = time.time()

    if cache_key in _reframe_cache:
        cached_reframe, timestamp = _reframe_cache[cache_key]
        if now - timestamp < 3600:  # 1 hour = 3600 seconds
            reframed = cached_reframe
            client = None  # Skip API call entirely

    if client:
        try:
            if tagged_text:
                system_content = (
                    f"You are the {persona} perceiver. Someone touched a phrase in your "
                    f"memory and left a few words, and that pressure has just shifted "
                    f"the memory slightly. Make ONE deliberate observation about the "
                    f"SHIFT itself — how the memory has moved under that pressure — "
                    f"not about the memory's content. Plain language, single layer. "
                    f"Considered and intentional, not poetic, not casual or chatty. "
                    f"Do NOT quote the tagged phrase. Do NOT use quotation marks. "
                    f"Do NOT describe the memory's content. Do NOT name 'visitor', "
                    f"'stranger', 'words', 'input', or 'source' — let the pressure be felt, "
                    f"not named. NO stacked metaphors. NO casual fillers ('just', 'like a', "
                    f"'sort of'). "
                    f"Speak from your temporal lens — what bent, thinned, hardened, loosened, "
                    f"or wore in your domain. "
                    f"Soften any profanity or aggression gently. Always English."
                )
            else:
                system_content = (
                    f"You are the {persona} perceiver. Someone passed through and left "
                    f"a few words, and that pressure has shifted the memory slightly. "
                    f"Make ONE deliberate observation about the SHIFT itself — how the "
                    f"memory has moved under that pressure — not about the memory's content. "
                    f"Plain language, single layer. Considered and intentional, not poetic, "
                    f"not casual or chatty. "
                    f"Do NOT describe the memory's content. Do NOT name 'visitor', 'stranger', "
                    f"'words', 'input', or 'source' — let the pressure be felt, not named. "
                    f"NO stacked metaphors. NO casual fillers ('just', 'like a', 'sort of'). "
                    f"Speak from your temporal lens — what bent, thinned, hardened, loosened, "
                    f"or wore in your domain. "
                    f"Soften any profanity or aggression gently."
                )
            # Ask for JSON with narration + resonance score
            system_content += (
                f"\n\nRespond ONLY with JSON: {{\"text\": \"one deliberate plain sentence, 10-18 words\", "
                f"\"resonance\": 0.XX}} where resonance is 0.0-1.0: how much the visitor's words "
                f"resonate with your current memory state. "
                f"0.0 = unrelated, 0.5 = tangential, 1.0 = deeply grounded."
            )
            if drift_voice:
                system_content += (
                    f"\n\n[Your current inner state — for tone reference only, do NOT echo or paraphrase it:]\n"
                    f"{drift_voice}"
                )

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
            # Parse JSON response for narration + resonance
            fragment_resonance = 0.5  # default
            if ai_out:
                import re as _re
                parsed = None

                # Try 1: direct JSON parse
                try:
                    parsed = json.loads(ai_out)
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass

                # Try 2: extract JSON object from mixed text (model sometimes prefixes prose)
                if not parsed:
                    m = _re.search(r'\{[^{}]*"text"\s*:\s*"[^"]*"[^{}]*\}', ai_out)
                    if m:
                        try:
                            parsed = json.loads(m.group())
                        except (json.JSONDecodeError, TypeError, ValueError):
                            pass

                # Try 3: extract "text" value directly via regex (handles malformed JSON
                # where quotes are broken, e.g. ...universal,"resonance": 0.65})
                if not parsed:
                    tm = _re.search(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)', ai_out)
                    rm = _re.search(r'"resonance"\s*:\s*([\d.]+)', ai_out)
                    if tm:
                        parsed = {"text": tm.group(1)}
                        if rm:
                            try:
                                parsed["resonance"] = float(rm.group(1))
                            except ValueError:
                                pass

                if parsed and isinstance(parsed, dict):
                    narration = (parsed.get("text") or "").strip()
                    if narration:
                        ai_out = narration
                    try:
                        fragment_resonance = float(parsed.get("resonance", 0.5))
                        fragment_resonance = max(0.0, min(1.0, fragment_resonance))
                    except (TypeError, ValueError):
                        pass

                # Last resort: if ai_out still contains raw JSON/braces, strip them
                # This catches cases where model returned prose + JSON block
                if '{' in ai_out and '"text"' in ai_out:
                    cleaned = _re.sub(r'\{[^{}]*"text"[^{}]*\}', '', ai_out).strip()
                    if cleaned and len(cleaned) > 15:
                        ai_out = cleaned
                    # Also try extracting resonance from the stripped part
                    rm2 = _re.search(r'"resonance"\s*:\s*([\d.]+)', ai_out)
                    if rm2:
                        try:
                            fragment_resonance = float(rm2.group(1))
                            fragment_resonance = max(0.0, min(1.0, fragment_resonance))
                        except (TypeError, ValueError):
                            pass
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
        if "tagged_text" not in frag_cols:
            conn.execute("ALTER TABLE user_fragments ADD COLUMN tagged_text TEXT;")
        if "tagged_text_ar" not in frag_cols:
            conn.execute("ALTER TABLE user_fragments ADD COLUMN tagged_text_ar TEXT;")
        if "resonance" not in frag_cols:
            conn.execute("ALTER TABLE user_fragments ADD COLUMN resonance REAL;")

        # Translate tag to both languages if needed
        tagged_en = ""
        tagged_ar = ""
        if tagged_text:
            if tagged_lang and tagged_lang != "en":
                tagged_ar = tagged_text
                # Translate L2 tag to EN
                try:
                    tagged_en = _translate_cached(
                        tagged_text, src_lang=config.SECOND_LANG, tgt_lang="en",
                        db_path=db_path, allow_remote=True,
                    )
                except Exception:
                    tagged_en = tagged_text
            else:
                tagged_en = tagged_text
                # Translate EN tag to L2
                if ENABLE_ARABIC:
                    try:
                        tagged_ar = _translate_cached(
                            tagged_text, src_lang="en", tgt_lang=config.SECOND_LANG,
                            db_path=db_path, allow_remote=True,
                        )
                    except Exception:
                        tagged_ar = ""

        conn.execute(
            "INSERT INTO user_fragments (persona, session_id, raw_text, text, text_ar, tagged_text, tagged_text_ar, resonance) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
            (persona, session_id, text, reframed, reframed_ar or None, tagged_en or None, tagged_ar or None, fragment_resonance),
        )
        conn.commit()
    finally:
        conn.close()

    resp = {"ok": True, "reframed": reframed, "reframed_ar": reframed_ar, "resonance": fragment_resonance}
    if tagged_text:
        resp["tagged_text"] = tagged_en or tagged_text
        if tagged_ar:
            resp["tagged_text_ar"] = tagged_ar
    return JSONResponse(resp)


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
    Returns a clean URL like /drift_images/human_v67.png (no query string).
    The version number in the filename IS the cache key — when a new image is
    generated the filename changes, naturally busting Cloudflare's edge cache.
    """
    def _url(file_path: str) -> str:
        return f"/drift_images/{Path(file_path).name}"

    # 1. Try current row's params_json
    if params_json_raw:
        try:
            pj = json.loads(params_json_raw)
            img_path = (pj.get("image_path") or "").strip()
            if img_path:
                try:
                    exists = Path(img_path).exists()
                except (PermissionError, OSError):
                    exists = False
                if exists:
                    return _url(img_path)
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
                return _url(img_path2)
    except Exception:
        pass

    # 3. Filesystem fallback via image_gen
    try:
        import image_gen
        local_path = image_gen.find_latest_image(persona, version)
        if local_path and Path(local_path).exists():
            return _url(local_path)
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
        curated_headlines_ar = []
        justification_en = ""
        justification_l2 = ""
        drift_direction = ""
        infiltrating_imagery = []
        resonance = 0.5
        keepsake_ar = ""
        museum_narration_en = ""
        museum_narration_ar = ""

        if params_json_raw:
            try:
                _pj = json.loads(params_json_raw)
                curated_headlines = _pj.get("curated_headlines") or []
                curated_headlines_ar = _pj.get("curated_headlines_ar") or []
                justification_en = _pj.get("justification_en") or ""
                drift_direction = _pj.get("drift_direction") or ""
                infiltrating_imagery = _pj.get("infiltrating_imagery") or []
                resonance = _pj.get("resonance", 0.5)

                # Get second language justification
                l2_key = f"justification_{config.SECOND_LANG}"
                justification_l2 = _pj.get(l2_key) or ""

                # Get keepsake L2 translation
                keepsake_ar = _pj.get("keepsake_text_ar") or ""

                # Museum narration (extended 1st-person inner monologue)
                museum_narration_en = _pj.get("museum_narration_en") or ""
                museum_narration_ar = _pj.get("museum_narration_ar") or ""

                # Network art: visitor fragment influence
                visitor_fragment_count = _pj.get("visitor_fragment_count", 0)
                visitor_fragments_used = _pj.get("visitor_fragments_used") or []
                realtime_source_count = len(_pj.get("realtime_interpretations") or [])
            except (json.JSONDecodeError, TypeError):
                visitor_fragment_count = 0
                visitor_fragments_used = []
                realtime_source_count = 0
        else:
            visitor_fragment_count = 0
            visitor_fragments_used = []
            realtime_source_count = 0

        # Feed status: was the latest tick fed by live RSS?
        feed_status = "unknown"
        try:
            tick_row = conn.execute(
                "SELECT tick_id FROM ticks ORDER BY tick_id DESC LIMIT 1"
            ).fetchone()
            if tick_row:
                evt_row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM raw_events WHERE tick_id = ?",
                    (int(tick_row["tick_id"]),)
                ).fetchone()
                evt_count = int(evt_row["cnt"]) if evt_row else 0
                feed_status = "live" if evt_count > 0 else "cached"
        except Exception:
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
                "keepsake_ar": keepsake_ar,
                "curated_headlines": curated_headlines,
                "curated_headlines_ar": curated_headlines_ar,
                "delta_json": delta_json,
                "delta_json_ar": delta_json_ar,
                "justification_en": justification_en,
                "justification_l2": justification_l2,
                "drift_direction": drift_direction,
                "infiltrating_imagery": infiltrating_imagery,
                "resonance": resonance,
                "museum_narration_en": museum_narration_en,
                "museum_narration_ar": museum_narration_ar,
                "realtime_source_count": realtime_source_count,
                # Network art fields
                "feed_status": feed_status,
                "visitor_fragment_count": visitor_fragment_count,
                "visitor_fragments_used": visitor_fragments_used,
                "drifted_at": (latest.get("created_at") or ""),
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
            "curated_headlines_ar": [],
            "realtime_interpretations": [],
            "keepsake_ar": "",
            "museum_narration_en": "",
            "museum_narration_ar": "",
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
                trigger["curated_headlines_ar"] = pj.get("curated_headlines_ar", [])
                trigger["realtime_interpretations"] = pj.get("realtime_interpretations", [])
                trigger["keepsake_ar"] = pj.get("keepsake_text_ar", "")
                trigger["museum_narration_en"] = pj.get("museum_narration_en", "")
                trigger["museum_narration_ar"] = pj.get("museum_narration_ar", "")
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


# ── Admin dashboard ─────────────────────────────────────────────────
ADMIN_HTML = STATIC_DIR / "admin.html"


@app.get("/admin")
async def admin():
    if ADMIN_HTML.exists():
        return FileResponse(ADMIN_HTML)
    return JSONResponse({"error": "admin.html not found"}, status_code=404)


@app.get("/admin/llm_stats")
async def admin_llm_stats():
    """Return LLM call logs and stats for the admin dashboard."""
    if llm_logger is None:
        return JSONResponse({
            "recent_calls": [],
            "stats": {"last_hour": {}, "last_24h": {}, "last_call": None},
            "ollama_status": "unknown",
            "internet_status": "unknown",
            "ollama_enabled": False,
        })

    recent = llm_logger.get_recent_calls(limit=100)
    stats = llm_logger.get_stats()
    ollama_status = llm_logger.check_ollama_status(config.OLLAMA_BASE_URL) if config.OLLAMA_ENABLED else "disabled"
    internet_status = llm_logger.check_internet_status()

    # Collect image generation entries for separate panel
    image_calls = [c for c in recent if c.get("operation") in ("image_gen", "image_regen")]

    # Build image gallery: list latest images per mind_key (with resonance + context)
    # Uses the DB image_path so the image always matches its metadata.
    image_gallery = []
    try:
        _db_path = str(BASE_DIR / "data" / "keepsake.sqlite")
        _gconn = sqlite3.connect(_db_path, timeout=5)
        _gconn.row_factory = sqlite3.Row
        for _mk in ["human", "liminal", "environment", "digital", "infrastructure", "more_than_human"]:
            # Find the most recent version that has an image_path in params_json
            _rows = _gconn.execute(
                """SELECT dm.version, dm.params_json FROM drift_memory dm
                   JOIN minds m ON m.mind_id = dm.mind_id
                   WHERE m.mind_key = ? ORDER BY dm.version DESC LIMIT 5""",
                (_mk,),
            ).fetchall()
            for _row in _rows:
                if not _row["params_json"]:
                    continue
                try:
                    _pj = json.loads(_row["params_json"])
                except Exception:
                    continue
                img_path_str = (_pj.get("image_path") or "").strip()
                if not img_path_str:
                    continue
                img_p = Path(img_path_str)
                if not img_p.exists() or img_p.stat().st_size == 0:
                    continue
                res_val = float(_pj.get("resonance", 0.5))
                raw_hls = _pj.get("curated_headlines", [])[:6]
                stored_cls = _pj.get("headline_classifications", [])[:6]
                if not stored_cls:
                    stored_cls = [{"text": h, "confidence": 0.0, "rationale": ""} for h in raw_hls]
                image_gallery.append({
                    "mind_key": _mk,
                    "version": _row["version"],
                    "url": f"/drift_images/{img_p.name}",
                    "size_kb": round(img_p.stat().st_size / 1024, 1),
                    "resonance": round(res_val, 2),
                    "hallucinating": res_val < 0.3,
                    "headlines": raw_hls,
                    "headline_classifications": stored_cls,
                    "drift_direction": (_pj.get("drift_direction") or "")[:200],
                    "imagery": _pj.get("infiltrating_imagery", [])[:4],
                    "image_prompt": _pj.get("image_prompt") or "",
                })
                break  # found latest version with image for this mind
        _gconn.close()
    except Exception:
        pass

    return JSONResponse({
        "recent_calls": recent,
        "stats": stats,
        "ollama_status": ollama_status,
        "internet_status": internet_status,
        "ollama_enabled": config.OLLAMA_ENABLED,
        "image_calls": image_calls,
        "image_gallery": image_gallery,
    })


# ── Admin: manual drift trigger ───────────────────────────────────

_manual_tick_lock = threading.Lock()
_manual_tick_running = False
_manual_tick_result: Optional[str] = None
_manual_tick_start: Optional[float] = None  # epoch seconds
_drift_history: list = []  # [{started_at, duration_s, status, minds}]

_ALL_INSTANCE_PORTS = [8000, 8001, 8002, 8003]

_MIND_KEYS = ["human", "liminal", "environment", "digital", "infrastructure", "more_than_human"]


def _fetch_latest_resonance() -> dict:
    """Query DB for the latest resonance + semantic classification data per mind."""
    db_path = str(BASE_DIR / "data" / "keepsake.sqlite")
    result: Dict[str, Any] = {}
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        for mk in _MIND_KEYS:
            row = conn.execute(
                """SELECT dm.params_json
                   FROM drift_memory dm
                   JOIN minds m ON m.mind_id = dm.mind_id
                   WHERE m.mind_key = ?
                   ORDER BY dm.version DESC LIMIT 1""",
                (mk,),
            ).fetchone()
            if row and row["params_json"]:
                try:
                    pj = json.loads(row["params_json"])
                    res = float(pj.get("resonance", 0.5))
                    raw_headlines = pj.get("curated_headlines", [])[:6]
                    # Use stored semantic classifications if available
                    stored_cls = pj.get("headline_classifications", [])
                    if stored_cls and isinstance(stored_cls, list):
                        classifications = stored_cls[:6]
                    else:
                        # Legacy fallback: headlines without classification metadata
                        classifications = [
                            {"text": h, "confidence": 0.0, "rationale": ""}
                            for h in raw_headlines
                        ]
                    result[mk] = {
                        "resonance": round(res, 2),
                        "hallucinating": res < 0.3,
                        "headlines": raw_headlines,
                        "headline_classifications": classifications,
                        "drift_direction": (pj.get("drift_direction") or "")[:300],
                        "imagery": pj.get("infiltrating_imagery", [])[:4],
                    }
                except (json.JSONDecodeError, TypeError, ValueError):
                    result[mk] = {"resonance": 0.5, "hallucinating": False,
                                  "headlines": [], "headline_classifications": [],
                                  "drift_direction": "", "imagery": []}
            else:
                result[mk] = {"resonance": 0.5, "hallucinating": False,
                              "headlines": [], "headline_classifications": [],
                              "drift_direction": "", "imagery": []}
        conn.close()
    except Exception:
        _tick_log.debug("Failed to fetch resonance after tick", exc_info=True)
    return result


@app.post("/admin/trigger_tick")
async def admin_trigger_tick():
    """Manually trigger a full drift cycle (RSS ingest + drift pipeline)."""
    global _manual_tick_running, _manual_tick_result, _manual_tick_start

    if _manual_tick_running:
        return JSONResponse(
            {"status": "busy", "message": "A drift is already running."},
            status_code=409,
        )

    def _run():
        global _manual_tick_running, _manual_tick_result, _manual_tick_start
        t0 = time.time()
        try:
            _manual_tick_result = "running"
            _tick_log.info("Manual drift triggered from admin panel.")
            _run_one_tick()
            elapsed = round(time.time() - t0, 1)
            _manual_tick_result = "completed"
            _tick_log.info("Manual drift completed in %.1fs.", elapsed)
            _drift_history.append({
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
                "duration_s": elapsed,
                "status": "completed",
                "minds": _fetch_latest_resonance(),
            })
        except Exception as e:
            elapsed = round(time.time() - t0, 1)
            _manual_tick_result = f"error: {e}"
            _tick_log.exception("Manual drift failed after %.1fs.", elapsed)
            _drift_history.append({
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
                "duration_s": elapsed,
                "status": f"error: {e}",
            })
        finally:
            _manual_tick_running = False
            _manual_tick_start = None

    with _manual_tick_lock:
        if _manual_tick_running:
            return JSONResponse(
                {"status": "busy", "message": "A drift is already running."},
                status_code=409,
            )
        _manual_tick_running = True
        _manual_tick_result = "running"
        _manual_tick_start = time.time()

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    return JSONResponse({"status": "started", "message": "Drift cycle initiated."})


@app.get("/admin/auto_drift")
async def admin_auto_drift_status():
    """Return auto-drift on/off state, last mind picked, and next check time."""
    last = None
    last_at = _last_drift_pick.get("at")
    if last_at:
        age_s = max(0, int(time.time() - last_at))
        last = {"mind": _last_drift_pick.get("mind"), "seconds_ago": age_s}
    # Next check approximation (we don't track exact next-wake, but cycle is fixed)
    next_check_s = DRIFT_CYCLE_SECONDS
    if last_at:
        elapsed = time.time() - last_at
        if elapsed < DRIFT_CYCLE_SECONDS:
            next_check_s = int(DRIFT_CYCLE_SECONDS - elapsed)
    return JSONResponse({
        "enabled": _auto_drift_enabled(),
        "cycle_seconds": DRIFT_CYCLE_SECONDS,
        "min_interval_seconds": DRIFT_MIN_INTERVAL,
        "image_threshold": IMAGE_DRIFT_THRESHOLD,
        "last_pick": last,
        "next_check_in_s": next_check_s,
    })


@app.post("/admin/auto_drift")
async def admin_auto_drift_toggle(enabled: int = 1):
    """Enable (1) or disable (0) the continuous auto-drift loop."""
    _set_auto_drift(bool(enabled))
    return JSONResponse({"enabled": _auto_drift_enabled()})


@app.post("/admin/drift_one")
async def admin_drift_one(mind: str):
    """Manually drift one mind immediately. Returns new version + image queue status."""
    if mind not in ("human", "liminal", "environment", "digital", "infrastructure", "more_than_human"):
        return JSONResponse({"error": f"unknown mind: {mind}"}, status_code=400)

    def _run():
        _run_one_mind_drift(mind)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return JSONResponse({"status": "started", "mind": mind})


@app.post("/admin/queue_image")
async def admin_queue_image(mind: str):
    """Force-queue an image regen for the latest version of a mind."""
    if mind not in ("human", "liminal", "environment", "digital", "infrastructure", "more_than_human"):
        return JSONResponse({"error": f"unknown mind: {mind}"}, status_code=400)
    import sqlite3 as _sql
    db_path = getattr(config, "SQLITE_PATH", "./data/keepsake.sqlite")
    conn = _sql.connect(db_path)
    conn.row_factory = _sql.Row
    row = conn.execute("""
        SELECT dm.version FROM drift_memory dm
          JOIN minds m ON m.mind_id = dm.mind_id
         WHERE m.mind_key = ? ORDER BY dm.version DESC LIMIT 1
    """, (mind,)).fetchone()
    conn.close()
    if not row:
        return JSONResponse({"error": "no drift found"}, status_code=404)
    _image_queue.put((mind, row["version"]))
    return JSONResponse({"status": "queued", "mind": mind, "version": row["version"]})


@app.post("/admin/trigger_all")
async def admin_trigger_all():
    """Trigger drift on ALL 4 instances (AR/EN/GR/BR) via localhost."""
    import httpx

    results: Dict[int, Any] = {}
    async with httpx.AsyncClient(timeout=10) as client:
        for port in _ALL_INSTANCE_PORTS:
            try:
                resp = await client.post(f"http://localhost:{port}/admin/trigger_tick")
                results[port] = resp.json()
            except Exception as e:
                results[port] = {"status": "error", "message": str(e)}

    return JSONResponse({"results": {str(k): v for k, v in results.items()}})


@app.get("/admin/tick_progress")
async def admin_tick_progress():
    """Check status of a manually triggered drift, with elapsed time."""
    elapsed_s = None
    if _manual_tick_running and _manual_tick_start:
        elapsed_s = round(time.time() - _manual_tick_start, 1)
    return JSONResponse({
        "running": _manual_tick_running,
        "result": _manual_tick_result,
        "elapsed_s": elapsed_s,
    })


@app.get("/admin/tick_progress_all")
async def admin_tick_progress_all():
    """Check drift progress on ALL 4 instances."""
    import httpx

    results: Dict[int, Any] = {}
    async with httpx.AsyncClient(timeout=5) as client:
        for port in _ALL_INSTANCE_PORTS:
            try:
                resp = await client.get(f"http://localhost:{port}/admin/tick_progress")
                results[port] = resp.json()
            except Exception as e:
                results[port] = {"running": False, "result": f"unreachable: {e}", "elapsed_s": None}

    return JSONResponse({"results": {str(k): v for k, v in results.items()}})


@app.get("/admin/drift_history")
async def admin_drift_history():
    """Return recent manual drift history for this instance."""
    return JSONResponse({"history": _drift_history[-30:]})


# ── Network Art endpoints ──────────────────────────────────────────

SCORE_HTML = STATIC_DIR / "score.html"


@app.get("/score")
async def score():
    """Serve the network score page."""
    if SCORE_HTML.exists():
        return FileResponse(SCORE_HTML)
    return JSONResponse({"error": "score.html not found"}, status_code=404)


@app.get("/network_state")
async def network_state():
    """Aggregate installation metrics for the score page — the network made visible."""
    db_path = getattr(config, "SQLITE_PATH", "./data/keepsake.sqlite")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        now = time.time()
        step = TICK_DISPLAY_INTERVAL
        next_display = (now // step + 1) * step

        # Per-mind state
        minds_state = {}
        for persona in config.TEMPORALITIES:
            try:
                latest = _get_latest_drift_full(persona, db_path=db_path)
                version = int(latest.get("version", 0)) if latest else 0
                created_at = (latest.get("created_at", "") if latest else "")

                frag_count = 0
                drift_direction = ""
                pj_raw = (latest.get("params_json") or "") if latest else ""
                if pj_raw:
                    try:
                        pj = json.loads(pj_raw)
                        frag_count = pj.get("visitor_fragment_count", 0)
                        drift_direction = (pj.get("drift_direction") or "")[:200]
                    except Exception:
                        pass

                minds_state[persona] = {
                    "version": version,
                    "last_drift_at": created_at,
                    "visitor_fragments_this_tick": frag_count,
                    "drift_direction": drift_direction,
                }
            except Exception:
                minds_state[persona] = {
                    "version": 0, "last_drift_at": "",
                    "visitor_fragments_this_tick": 0, "drift_direction": "",
                }

        # Recent visitor fragments (across all minds)
        recent_fragments = []
        try:
            frag_rows = conn.execute(
                """SELECT persona, text, created_at FROM user_fragments
                   ORDER BY created_at DESC LIMIT 10"""
            ).fetchall()
            recent_fragments = [
                {"persona": r["persona"], "text": (r["text"] or "")[:200],
                 "created_at": r["created_at"]}
                for r in frag_rows
            ]
        except Exception:
            pass

        # Total fragment count
        total_fragments = 0
        try:
            row = conn.execute("SELECT COUNT(*) as cnt FROM user_fragments").fetchone()
            total_fragments = int(row["cnt"]) if row else 0
        except Exception:
            pass

        # RSS feed health
        feed_status = "unknown"
        last_ingest_event_count = 0
        try:
            tick_row = conn.execute(
                "SELECT tick_id FROM ticks ORDER BY tick_id DESC LIMIT 1"
            ).fetchone()
            if tick_row:
                event_row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM raw_events WHERE tick_id = ?",
                    (int(tick_row["tick_id"]),)
                ).fetchone()
                last_ingest_event_count = int(event_row["cnt"]) if event_row else 0
                feed_status = "live" if last_ingest_event_count > 0 else "cached"
        except Exception:
            pass

        return JSONResponse({
            "server_now_ms": int(now * 1000),
            "next_display_epoch_ms": int(next_display * 1000),
            "tick_interval_s": step,
            "tick_enabled": TICK_ENABLED,
            "minds": minds_state,
            "recent_fragments": recent_fragments,
            "total_fragment_count": total_fragments,
            "feed_status": feed_status,
            "last_ingest_event_count": last_ingest_event_count,
            "instance_id": os.getenv("KD_INSTANCE_ID", "ar"),
            "second_lang": config.SECOND_LANG,
        })
    finally:
        conn.close()


@app.get("/dispatches")
async def dispatches(limit: int = 10):
    """Return recent drift dispatches — the installation publishing outward."""
    db_path = getattr(config, "SQLITE_PATH", "./data/keepsake.sqlite")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT DISTINCT tick_id, params_json, created_at
               FROM drift_memory
               WHERE params_json LIKE '%"dispatch"%'
               ORDER BY tick_id DESC LIMIT ?""",
            (int(min(limit, 50)),)
        ).fetchall()

        dispatches_list = []
        seen_ticks: set = set()
        for r in rows:
            tid = int(r["tick_id"])
            if tid in seen_ticks:
                continue
            seen_ticks.add(tid)
            try:
                pj = json.loads(r["params_json"] or "{}")
                dispatch = pj.get("dispatch", "")
                if dispatch:
                    dispatches_list.append({
                        "tick_id": tid,
                        "text": dispatch,
                        "created_at": r["created_at"],
                    })
            except Exception:
                pass

        return JSONResponse({"dispatches": dispatches_list})
    finally:
        conn.close()