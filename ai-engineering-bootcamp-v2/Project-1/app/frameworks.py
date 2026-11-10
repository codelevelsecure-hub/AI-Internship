"""The NIST AI RMF subcategory list (frameworks/nist_ai_rmf_1.0.json) — the source of truth for IDs."""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

FRAMEWORK_FILE = Path(__file__).resolve().parent.parent / "frameworks" / "nist_ai_rmf_1.0.json"


@dataclass(frozen=True)
class Framework:
    name: str
    version: str
    functions: list[str]
    categories: list[dict[str, Any]]
    subcategories: list[dict[str, Any]]

    def subcategory(self, subcategory_id: str) -> dict[str, Any]:
        for sub in self.subcategories:
            if sub["id"] == subcategory_id:
                return sub
        raise KeyError(f"Unknown subcategory: {subcategory_id}")

    def category(self, category_id: str) -> dict[str, Any]:
        for cat in self.categories:
            if cat["id"] == category_id:
                return cat
        raise KeyError(f"Unknown category: {category_id}")

    def ids(self) -> list[str]:
        return [sub["id"] for sub in self.subcategories]


def normalize_id(raw: str) -> str:
    """'govern 1.1', 'GOVERN-1.1', ' Govern 1.1 ' -> 'GOVERN 1.1'."""

    return " ".join(raw.replace("-", " ").upper().split())


@lru_cache(maxsize=1)
def load_framework() -> Framework:
    data = json.loads(FRAMEWORK_FILE.read_text(encoding="utf-8"))
    return Framework(
        name=data["framework"],
        version=data["version"],
        functions=data["functions"],
        categories=data["categories"],
        subcategories=data["subcategories"],
    )
