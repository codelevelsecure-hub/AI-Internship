"""Retrieve from both corpora with one question embedding, and label passages for citation.

Standards passages are labelled S1, S2, ... and organization passages O1, O2, ... (best first).
When the question names subcategories (e.g. "MEASURE 2.7"), passages tagged with those IDs are
fetched first in both namespaces, so an exact ID always reaches the prompt.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from app.frameworks import load_framework, normalize_id
from app.retrieval import embed_query, get_settings, query_by_vector

SUBCATEGORY_PATTERN = re.compile(r"\b(GOVERN|MAP|MEASURE|MANAGE)[\s-]*(\d+)\.(\d+)\b", re.IGNORECASE)
MAX_NAMED_IDS = 3  # Bound the extra queries a question can trigger.
MAPPING_DOC_TYPES = ("crosswalk", "control_mapping")
MAPPING_SLOTS = 2


@dataclass
class Retrieved:
    standards: list[dict[str, Any]] = field(default_factory=list)
    organization: list[dict[str, Any]] = field(default_factory=list)
    named_subcategories: list[str] = field(default_factory=list)
    embedding_tokens: int = 0

    def all(self) -> list[dict[str, Any]]:
        return self.standards + self.organization

    def by_label(self) -> dict[str, dict[str, Any]]:
        return {m["label"]: m for m in self.all()}


def named_subcategories(text: str) -> list[str]:
    """Valid NIST subcategory IDs mentioned in the text, in order, without duplicates."""

    known = set(load_framework().ids())
    found = [normalize_id(f"{m[1]} {m[2]}.{m[3]}") for m in SUBCATEGORY_PATTERN.finditer(text)]
    return list(dict.fromkeys(sid for sid in found if sid in known))


def _dedupe(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen, out = set(), []
    for m in matches:
        if m["id"] not in seen:
            seen.add(m["id"])
            out.append(m)
    return out


def search_both(question: str, top_k: int = 5) -> Retrieved:
    settings = get_settings()
    named = named_subcategories(question)[:MAX_NAMED_IDS]
    # An ID alone ("MEASURE 2.7") carries no meaning for embeddings; add what the subcategory says,
    # so organization documents about the same topic (e.g. security reviews) are found too.
    framework = load_framework()
    search_text = "\n".join([question] + [f"{sid}: {framework.subcategory(sid)['text']}" for sid in named])
    vector, tokens = embed_query(search_text)

    standards: list[dict[str, Any]] = []
    organization: list[dict[str, Any]] = []
    for sid in named:
        id_filter = {"subcategory_id": {"$eq": sid}}
        standards += query_by_vector(vector, 3, settings.standards_namespace, id_filter)
        organization += query_by_vector(vector, 2, settings.org_namespace, id_filter)
    standards += query_by_vector(vector, top_k, settings.standards_namespace)
    # Mapping records repeat NIST's wording, so they outscore real evidence; give evidence documents
    # their own top_k and let mapping records take at most MAPPING_SLOTS more places.
    organization += query_by_vector(vector, top_k, settings.org_namespace, {"doc_type": {"$nin": list(MAPPING_DOC_TYPES)}})
    organization += query_by_vector(vector, MAPPING_SLOTS, settings.org_namespace, {"doc_type": {"$in": list(MAPPING_DOC_TYPES)}})

    result = Retrieved(_dedupe(standards), _dedupe(organization), named, tokens)
    for prefix, matches in (("S", result.standards), ("O", result.organization)):
        for i, m in enumerate(matches, start=1):
            m["label"] = f"{prefix}{i}"
            m["corpus"] = "standard" if prefix == "S" else "organization"
    return result
