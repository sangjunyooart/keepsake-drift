# models.py
from __future__ import annotations

from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class Segment(BaseModel):
    id: str
    temporality: str
    change: str
    text_en: str
    text_ar: Optional[str] = ""


class ChatRequest(BaseModel):
    session_id: str
    persona: str
    message: str
    lang: str = "en"


class ChatUIResponse(BaseModel):
    """
    Web UI response (v1.65)

    Minimal stable contract:
    - reply_en (always)
    - reply_ar (optional string)
    """
    reply_en: str
    reply_ar: str = ""


class StateResponse(BaseModel):
    """
    UI polling contract (v1.65)
    """
    temporality: str
    version: int
    image_url: Optional[str] = None
    drift_en: str = ""
    drift_ar: str = ""
    recap_en: str = ""
    recap_ar: str = ""


class ChatResponse(BaseModel):
    """
    Legacy / internal richer response (kept for compatibility).
    Not required by the web UI endpoint in v1.65, but preserved.
    """
    reply_en: str
    reply_ar: str = ""
    segments: List[Segment] = []
    meta: Dict[str, Any] = {}