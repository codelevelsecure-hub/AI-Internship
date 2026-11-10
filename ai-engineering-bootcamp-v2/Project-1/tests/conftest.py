"""Shared test setup."""

import pytest


@pytest.fixture(autouse=True)
def no_api_key_by_default(monkeypatch):
    """Tests run with auth off unless a test sets API_KEY itself (a real key in .env must not break them)."""

    monkeypatch.delenv("API_KEY", raising=False)
