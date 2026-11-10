"""Offline tests for the X-API-Key check."""

from fastapi.testclient import TestClient

import app.api as api

client = TestClient(api.app)


def _no_pinecone(monkeypatch):
    monkeypatch.setattr(api, "pinecone_health", lambda: {"ok": True})


def test_open_when_no_key_configured(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    _no_pinecone(monkeypatch)
    assert client.post("/ask/simple", json={"question": " "}).status_code == 400  # reached the route
    assert client.get("/health/pinecone").json()["auth_enabled"] is False


def test_key_required_everywhere_except_health_and_docs(monkeypatch):
    monkeypatch.setenv("API_KEY", "s3cret")
    _no_pinecone(monkeypatch)
    assert client.post("/ask", json={"question": "x"}).status_code == 401
    assert client.get("/controls").status_code == 401
    assert client.post("/ask/simple", json={"question": " "}, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/ask/simple", json={"question": " "}, headers={"X-API-Key": "s3cret"}).status_code == 400
    health = client.get("/health/pinecone")
    assert health.status_code == 200 and health.json()["auth_enabled"] is True
    assert client.get("/docs").status_code == 200 and client.get("/openapi.json").status_code == 200
