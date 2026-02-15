# keepsake_db.py
import sqlite3
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "keepsake_memories.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    기존 memory 테이블 + 새 chat_messages 테이블을 한 번에 초기화.
    서버 시작할 때 한 번만 호출.
    """
    conn = get_conn()
    cur = conn.cursor()

    # (이미 쓰고 있던 기억 테이블이 있다면 유지)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            persona TEXT NOT NULL,
            temporality TEXT,
            text_en TEXT NOT NULL,
            text_ar TEXT,
            created_at TEXT NOT NULL
        );
        """
    )

    # 새로 추가: 채팅 히스토리 테이블
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            persona     TEXT NOT NULL,
            role        TEXT NOT NULL,  -- 'user' or 'ai'
            lang        TEXT NOT NULL,  -- 'en' or 'ar'
            text_en     TEXT,
            text_ar     TEXT,
            created_at  TEXT NOT NULL
        );
        """
    )

    conn.commit()
    conn.close()


# --- chat_messages 관련 함수들 ---

def save_chat_message(session_id: str,
                      persona: str,
                      role: str,
                      lang: str,
                      text_en: str | None,
                      text_ar: str | None) -> None:
    """
    한 줄의 채팅 메시지를 DB에 저장.
    role: 'user' 또는 'ai'
    lang: 'en' 또는 'ar' (UI 기준)
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO chat_messages
            (session_id, persona, role, lang, text_en, text_ar, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
        (
            session_id,
            persona,
            role,
            lang,
            text_en,
            text_ar,
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def get_recent_chat_history(session_id: str,
                            persona: str,
                            limit: int = 8) -> list[dict]:
    """
    특정 session + persona에 대해 최근 limit개 (user+ai) 메시지를
    과거 -> 현재 순서로 리스트 반환.
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT role, text_en, text_ar
        FROM chat_messages
        WHERE session_id = ? AND persona = ?
        ORDER BY id DESC
        LIMIT ?;
        """,
        (session_id, persona, limit),
    )
    rows = cur.fetchall()
    conn.close()

    history: list[dict] = []
    # DESC로 가져왔으니 다시 뒤집어서 과거 -> 현재로
    for r in reversed(rows):
        history.append(
            {
                "role": r["role"],
                "en": r["text_en"] or "",
                "ar": r["text_ar"] or "",
            }
        )
    return history