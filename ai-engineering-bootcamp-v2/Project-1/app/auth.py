"""API-key check for every route except the health check and the API docs.

Set API_KEY in the environment (always on a deployed server); clients send it as the X-API-Key header.
With API_KEY unset the API is open — convenient locally, and /health/pinecone reports auth_enabled=false
so an unprotected deployment is easy to spot.
"""

import os
import secrets

from fastapi import Request
from fastapi.responses import JSONResponse

OPEN_PATHS = {"/health/pinecone", "/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"}
HEADER = "X-API-Key"


def configured_key() -> str:
    return os.getenv("API_KEY", "").strip()


def auth_enabled() -> bool:
    return bool(configured_key())


async def require_api_key(request: Request, call_next):
    expected = configured_key()
    if expected and request.url.path not in OPEN_PATHS:
        supplied = request.headers.get(HEADER, "")
        if not secrets.compare_digest(supplied.encode(), expected.encode()):
            return JSONResponse({"detail": f"Missing or invalid {HEADER} header."}, status_code=401)
    return await call_next(request)
