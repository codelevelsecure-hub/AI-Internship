"""The organization's control mapping (NIST subcategory <-> organization control), read from Pinecone.

Records are stored by ingest_docs.py as "crosswalk#<n>" chunks in the organization namespace, so the
laptop and a deployed server read the same mapping. Loaded once and cached; call reload() after ingest.
"""

from functools import lru_cache
from typing import Any

from app.frameworks import load_framework, normalize_id
from app.ingestion.control_mapping import CROSSWALK_DOCUMENT_ID
from app.retrieval import fetch_document_chunks, get_settings


@lru_cache(maxsize=1)
def load_records() -> tuple[dict[str, Any], ...]:
    chunks = fetch_document_chunks(CROSSWALK_DOCUMENT_ID, get_settings().org_namespace)
    return tuple(
        {
            "subcategory_id": c["metadata"]["subcategory_id"],
            "function": c["metadata"]["function"],
            "control_name": c["metadata"]["control_name"],
            "source": c["metadata"].get("source"),
            "chunk_id": c["id"],
        }
        for c in chunks
    )


def reload() -> None:
    load_records.cache_clear()


def controls_for(subcategory_id: str) -> list[dict[str, Any]]:
    sid = normalize_id(subcategory_id)
    return [r for r in load_records() if r["subcategory_id"] == sid]


def subcategories_for(control_name: str) -> list[dict[str, Any]]:
    wanted = control_name.strip().lower()
    return [r for r in load_records() if r["control_name"].lower() == wanted]


def controls() -> list[dict[str, Any]]:
    """One entry per control, with every subcategory it is mapped to."""

    grouped: dict[str, dict[str, Any]] = {}
    for r in load_records():
        entry = grouped.setdefault(r["control_name"], {"control_name": r["control_name"], "subcategory_ids": []})
        entry["subcategory_ids"].append(r["subcategory_id"])
    return list(grouped.values())


def mapping_summary(subcategory_ids: list[str]) -> list[str]:
    """Plain-language lines for the prompt, e.g. 'MEASURE 2.7: no organization control is mapped.'"""

    framework = load_framework()
    lines = []
    for sid in subcategory_ids:
        mapped = controls_for(sid)
        sub = framework.subcategory(sid)
        if mapped:
            names = ", ".join(f'"{m["control_name"]}"' for m in mapped)
            lines.append(f"{sid} ({sub['text']}) is mapped to organization control(s): {names}.")
        else:
            lines.append(f"{sid} ({sub['text']}) has NO organization control mapped to it.")
    return lines
