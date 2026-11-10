"""capabilities.json must describe exactly the API the app serves."""

import json
from pathlib import Path

from fastapi.routing import APIRoute

from app.api import app
from app.auth import OPEN_PATHS
from app.frameworks import load_framework

CAPABILITIES = json.loads((Path(__file__).resolve().parent.parent / "capabilities.json").read_text(encoding="utf-8"))


def test_every_route_is_described_and_every_description_is_a_route():
    served = {(m, r.path) for r in app.routes if isinstance(r, APIRoute) for m in r.methods}
    described = {(c["method"], c["path"]) for c in CAPABILITIES["capabilities"]}
    assert described == served


def test_capability_ids_unique_and_facts_match_the_code():
    ids = [c["id"] for c in CAPABILITIES["capabilities"]]
    assert len(ids) == len(set(ids))
    fw = CAPABILITIES["scope"]["frameworks"][0]
    assert fw["subcategories"] == len(load_framework().subcategories)
    assert fw["categories"] == len(load_framework().categories)
    assert set(CAPABILITIES["api"]["authentication"]["open_paths"]) <= OPEN_PATHS
