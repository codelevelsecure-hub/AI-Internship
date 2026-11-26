"""Parse the organization's AI RMF Control Mapping into crosswalk records.

The mapping file is a set of markdown tables, one per NIST function:
    | NIST AI RMF | Acu Control          |
    | GOVERN 1.1  | AI Governance Policy |
Each row becomes {"subcategory_id", "function", "control_name"}. IDs are checked against the
framework JSON, so a typo in the mapping fails loudly instead of silently dropping a control.
"""

import re
from pathlib import Path
from typing import Any

from app.frameworks import load_framework, normalize_id

CROSSWALK_DOCUMENT_ID = "crosswalk"  # chunk ids "crosswalk#<n>" in the organization namespace
ROW = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$")


def parse_control_mapping(path: Path) -> list[dict[str, Any]]:
    framework = load_framework()
    known = set(framework.ids())
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = ROW.match(line.strip())
        if not match:
            continue
        left, right = match.group(1), match.group(2)
        if set(left) <= set("-: ") or left.upper().startswith("NIST"):
            continue  # header or separator row
        subcategory_id = normalize_id(left)
        if subcategory_id not in known:
            raise ValueError(f"{path.name}: '{left}' is not a NIST AI RMF 1.0 subcategory")
        records.append({
            "subcategory_id": subcategory_id,
            "function": framework.subcategory(subcategory_id)["function"],
            "control_name": right,
        })
    if not records:
        raise ValueError(f"{path.name}: no mapping rows found")
    return records


def crosswalk_chunks(records: list[dict[str, Any]], organization: str, source: str) -> list[dict[str, Any]]:
    """One small, searchable chunk per mapping row; the metadata carries the structured record."""

    framework = load_framework()
    chunks = []
    for i, rec in enumerate(records):
        sub = framework.subcategory(rec["subcategory_id"])
        text = (
            f"{organization} control mapping: NIST AI RMF {rec['subcategory_id']} is addressed by the "
            f"{organization} control \"{rec['control_name']}\".\n"
            f"{rec['subcategory_id']}: {sub['text']}"
        )
        chunks.append({
            "id": f"{CROSSWALK_DOCUMENT_ID}#{i}",
            "text": text,
            "metadata": {
                "document_id": CROSSWALK_DOCUMENT_ID,
                "doc_type": "crosswalk",
                "corpus": "internal",
                "organization": organization,
                "source": source,
                "title": "AI RMF Control Mapping",
                "section": f"{rec['function']} mapping",
                "subcategory_id": rec["subcategory_id"],
                "function": rec["function"],
                "control_name": rec["control_name"],
                "chunk_index": i,
            },
        })
    return chunks
