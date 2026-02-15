import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"
GENERATED_DIR = STATIC_DIR / "generated"

DATA_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

SQLITE_PATH = str(DATA_DIR / "keepsake.sqlite")

# Canonical keys used everywhere (UI param, state endpoint, db rows)
TEMPORALITIES = [
    "human",
    "liminal",
    "environment",
    "digital",
    "infrastructure",
    "more_than_human",
]

# External original memories file (NOT in config)
ORIGINAL_MEMORIES_PATH = str(DATA_DIR / "original_memories.json")

# Optional second-language support (easy to disable for other deployments)
ENABLE_ARABIC = os.getenv("ENABLE_ARABIC", "0").strip().lower() in ("1", "true", "yes", "y")
ENABLE_TRANSLATION = ENABLE_ARABIC  # alias — backwards compat

# Which second language is active: "ar" (Arabic), "el" (Greek), "pt-br" (Brazilian Portuguese)
SECOND_LANG = os.getenv("KD_SECOND_LANG", "ar").strip().lower()
if SECOND_LANG not in ("ar", "el", "pt-br"):
    SECOND_LANG = "ar"

LANG_CONFIG = {
    "ar": {
        "name": "Arabic", "native_name": "العربية",
        "toggle_label": "عرب", "direction": "rtl",
        "rules": "Use Modern Standard Arabic (MSA / Fusha). Keep proper nouns as commonly written in Arabic.",
    },
    "el": {
        "name": "Greek", "native_name": "Ελληνικά",
        "toggle_label": "Ελλ", "direction": "ltr",
        "rules": "Use modern monotonic Greek orthography. Preserve accents (tonos). Transliterate proper nouns into Greek when a standard form exists.",
    },
    "pt-br": {
        "name": "Brazilian Portuguese", "native_name": "Português",
        "toggle_label": "PT", "direction": "ltr",
        "rules": "Use Brazilian Portuguese (not European Portuguese). Apply the 2009 orthographic agreement. Keep proper nouns in their original form unless a standard Portuguese form exists.",
    },
}

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", os.getenv("OPENAI_MODEL", "gpt-5.2"))

# Limits
MAX_VISITOR_CHARS = int(os.getenv("MAX_VISITOR_CHARS", "280"))

# Cultural safety mode
# - "" (default): normal
# - "dubai": strict anti-trigger posture (religion + profanity + sectarian debate deflection)
CULTURAL_SAFE_MODE = os.getenv("CULTURAL_SAFE_MODE", "").strip().lower()