"""Find the organization's evidence for a NIST subcategory or for one of its controls. No LLM call.

The search text combines the subcategory statement(s) with the mapped control names, and only
evidence documents are searched (policy, procedure, minutes, risk assessment, inventory) —
mapping records say what is claimed, not what is done.
"""

from typing import Any

from app import crosswalk
from app.frameworks import load_framework, normalize_id
from app.retrieval import embed_query, embedding_cost_usd, get_settings, query_by_vector
from app.search import MAPPING_DOC_TYPES

QUOTE_CHARS = 400


def _evidence_item(match: dict[str, Any]) -> dict[str, Any]:
    meta = match["metadata"]
    body = match["text"].split("\n", 1)[-1].strip()  # drop the "<org> — <title> — <section>" header line
    return {
        "document_id": meta.get("document_id"),
        "title": meta.get("title"),
        "doc_type": meta.get("doc_type"),
        "section": meta.get("section"),
        "date": meta.get("date"),
        "ai_system": meta.get("ai_system"),
        "chunk_id": match["id"],
        "score": round(match["score"], 4),
        "quote": body[:QUOTE_CHARS] + ("…" if len(body) > QUOTE_CHARS else ""),
    }


def _search_evidence(search_text: str, top_k: int) -> tuple[list[dict[str, Any]], int]:
    vector, tokens = embed_query(search_text)
    matches = query_by_vector(vector, top_k, get_settings().org_namespace,
                              {"doc_type": {"$nin": list(MAPPING_DOC_TYPES)}})
    return [_evidence_item(m) for m in matches], tokens


def evidence_for_subcategory(subcategory_id: str, top_k: int = 5) -> dict[str, Any]:
    """Raises KeyError for an unknown subcategory."""

    sub = load_framework().subcategory(normalize_id(subcategory_id))
    mapped = crosswalk.controls_for(sub["id"])
    control_names = [m["control_name"] for m in mapped]
    search_text = "\n".join([f"{sub['id']}: {sub['text']}"] + [f"Control: {name}" for name in control_names])
    evidence, tokens = _search_evidence(search_text, top_k)
    return {
        "subcategory": sub,
        "mapping_status": "mapped" if mapped else "unmapped",
        "mapped_controls": control_names,
        # Without a mapped control, matches are related by topic only — not claimed coverage.
        "evidence_basis": "mapped controls + topic similarity" if mapped else "topic similarity only (no mapped control)",
        "evidence": evidence,
        "embedding_tokens": tokens,
        "cost_usd": round(embedding_cost_usd(tokens), 8),
    }


def evidence_for_control(control_name: str, top_k: int = 5) -> dict[str, Any]:
    """Raises KeyError for a control that is not in the mapping."""

    records = crosswalk.subcategories_for(control_name)
    if not records:
        raise KeyError(f"Unknown control: {control_name}")
    framework = load_framework()
    subs = [framework.subcategory(r["subcategory_id"]) for r in records]
    name = records[0]["control_name"]
    search_text = "\n".join([f"Control: {name}"] + [f"{s['id']}: {s['text']}" for s in subs])
    evidence, tokens = _search_evidence(search_text, top_k)
    return {
        "control_name": name,
        "subcategories": subs,
        "evidence": evidence,
        "embedding_tokens": tokens,
        "cost_usd": round(embedding_cost_usd(tokens), 8),
    }
