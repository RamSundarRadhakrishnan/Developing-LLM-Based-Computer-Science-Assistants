"""Paths and provider settings; secrets stay server-side."""
import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent
DEMO_DIR = BACKEND_DIR.parent
REPOSITORY_ROOT = DEMO_DIR.parent
load_dotenv(REPOSITORY_ROOT / ".env", override=False)

CHAT_MODEL = "gemini-3.5-flash-lite"
JUDGE_MODEL = "nvidia/nemotron-3.5-content-safety:free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{CHAT_MODEL}:generateContent"
SESSION_TTL_SECONDS = 1800


def openrouter_key() -> str:
    return os.environ.get("OPENROUTER_KEY", "").strip()


def gemini_key() -> str:
    return os.environ.get("GEMINI_API_KEY", "").strip()
