# backend/api/config.py
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"

UPLOAD_DIR.mkdir(exist_ok=True)

CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]