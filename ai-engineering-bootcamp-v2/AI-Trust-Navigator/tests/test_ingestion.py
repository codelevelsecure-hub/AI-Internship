"""Offline tests for the framework file, the Control Mapping parser and the internal splitter."""

import pytest

from app.frameworks import load_framework, normalize_id
from app.ingestion.control_mapping import parse_control_mapping
from app.ingestion.internal_splitter import split_internal


def test_framework_has_72_unique_subcategories_in_19_categories():
    fw = load_framework()
    ids = fw.ids()
    assert len(ids) == 72 and len(set(ids)) == 72
    assert len(fw.categories) == 19
    assert {s["function"] for s in fw.subcategories} == {"Govern", "Map", "Measure", "Manage"}
    assert all(s["text"] and s["page"] for s in fw.subcategories)
    assert fw.subcategory("MEASURE 2.7")["category_id"] == "MEASURE 2"


def test_normalize_id():
    assert normalize_id(" govern-1.1 ") == "GOVERN 1.1"


def test_control_mapping_parses_rows_and_rejects_unknown_ids(tmp_path):
    good = tmp_path / "mapping.txt"
    good.write_text(
        "## Governance\n\n| NIST AI RMF | Acu Control |\n| --- | --- |\n| GOVERN 1.1 | AI Governance Policy |\n"
        "| map 1.1 | AI Inventory |\n",
        encoding="utf-8",
    )
    assert parse_control_mapping(good) == [
        {"subcategory_id": "GOVERN 1.1", "function": "Govern", "control_name": "AI Governance Policy"},
        {"subcategory_id": "MAP 1.1", "function": "Map", "control_name": "AI Inventory"},
    ]

    bad = tmp_path / "bad.txt"
    bad.write_text("| NIST AI RMF | Control |\n| --- | --- |\n| GOVERN 9.9 | Nothing |\n", encoding="utf-8")
    with pytest.raises(ValueError, match="GOVERN 9.9"):
        parse_control_mapping(bad)


def test_internal_splitter_sections_stop_at_and_metadata(tmp_path):
    doc = tmp_path / "Inventory.txt"
    doc.write_text(
        "AI System Inventory\n\n## System 1\n\n### Chatbot\n\nRisk Rating:\nMedium\n" + "Details. " * 20 +
        "\n\n---\n\n## System 2\n\n### Scorer\n\nRisk Rating:\nHigh\n" + "More. " * 30 +
        "\n\n# Document 3 – Something else\n\nshould not appear\n",
        encoding="utf-8",
    )
    entry = {"document_id": "inv", "doc_type": "system_inventory", "stop_at": "# Document 3"}
    chunks = split_internal(doc, entry, "Acu")
    assert [c["id"] for c in chunks] == ["inv#0", "inv#1"]
    assert [c["metadata"]["ai_system"] for c in chunks] == ["Chatbot", "Scorer"]
    assert all(c["metadata"]["corpus"] == "internal" for c in chunks)
    assert not any("should not appear" in c["text"] for c in chunks)
    assert chunks[0]["text"].startswith("Acu — AI System Inventory — System 1")
