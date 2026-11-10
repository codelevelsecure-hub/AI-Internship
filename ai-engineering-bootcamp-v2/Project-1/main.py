"""Entry point for AI Trust Navigator. The application lives in the app/ package.

Run from this folder:  uvicorn main:app --host 127.0.0.1 --port 8001 --reload
"""

from app.api import app  # noqa: F401
