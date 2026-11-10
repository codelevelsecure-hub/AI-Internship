"""Split the NIST AI RMF PDFs along the framework's own structure.

AI RMF 100-1 produces:
  - one chunk per subcategory statement, taken from frameworks/nist_ai_rmf_1.0.json (clean text);
  - prose chunks from the rest of the document, with the category tables removed
    (their two-column layout extracts as interleaved fragments, and the JSON already holds them).
The Playbook produces chunks per subcategory and section: About, Suggested Actions,
Transparency & Documentation. References are skipped (citation lists retrieve poorly).

Every chunk is {"id", "text", "metadata"}; ids are "<document_id>#<n>" so re-ingest replaces cleanly.
"""

import re
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.frameworks import load_framework
from app.ingestion.pdf import Page, read_pdf_pages

FUNCS = "GOVERN|MAP|MEASURE|MANAGE"
CHUNK_SIZE, CHUNK_OVERLAP = 1200, 150

# AI RMF 100-1 layout
FRAMEWORK_SKIP_PAGES = 5  # cover, legal notice, contents, list of figures
FRAMEWORK_NOISE = re.compile(r"^(NIST AI 100-1|AI RMF 1\.0|Page \d+|Categories|Subcategories|Continued on next page)\s*$")
TABLE_START = re.compile(r"^Table \d+: Categories and subcategories for the")
SECTION_NUMBER = re.compile(r"^\d+\.(\d+)?$")  # "5.2", "6." — a category table always ends right before one

# Playbook layout
PLAYBOOK_HEADING = re.compile(rf"^({FUNCS}) (\d+\.\d+)$")
PLAYBOOK_NOISE = re.compile(r"^\d+ of \d+$")
PLAYBOOK_SECTIONS = {
    "About": "about",
    "Suggested Actions": "suggested_actions",
    "Transparency & Documentation": "documentation",
    "AI Transparency Resources": "documentation",
    "References": "references",
}
SECTION_LABELS = {"statement": "Statement", "about": "About", "suggested_actions": "Suggested Actions", "documentation": "Transparency & Documentation"}


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def _clean(lines: list[str]) -> str:
    """Join PDF lines: glue lone bullet markers to their text, undo end-of-line hyphenation, keep paragraphs."""

    text = "\n".join(line.rstrip() for line in lines)
    text = re.sub(r"•\s*\n\s*", "• ", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # "manage-\nment" -> "management"
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def split_framework(path: Path, entry: dict[str, Any]) -> list[dict[str, Any]]:
    """AI RMF 100-1 -> subcategory statement chunks + prose chunks."""

    doc_id, title = entry["document_id"], entry["title"]
    framework = load_framework()
    base = {"document_id": doc_id, "doc_type": entry["doc_type"], "source": path.name, "title": title, "corpus": "external"}
    chunks: list[dict[str, Any]] = []

    # 1) One chunk per subcategory, from the checked JSON.
    for sub in framework.subcategories:
        cat = framework.category(sub["category_id"])
        text = (
            f"NIST AI RMF {sub['id']} ({sub['function']} function)\n"
            f"Category {cat['id']}: {cat['text']}\n"
            f"Subcategory {sub['id']}: {sub['text']}"
        )
        meta = {**base, "section_type": "subcategory", "subcategory_id": sub["id"],
                "category_id": sub["category_id"], "function": sub["function"], "page": sub["page"]}
        chunks.append({"text": text, "metadata": meta})

    # 2) Prose pages, tables removed.
    in_table = False
    splitter = _splitter()
    for page in read_pdf_pages(path):
        if page.number <= FRAMEWORK_SKIP_PAGES:
            continue
        kept: list[str] = []
        for raw in page.text.splitlines():
            line = raw.strip()
            if TABLE_START.match(line):
                in_table = True
            elif in_table and SECTION_NUMBER.match(line):
                in_table = False
            if in_table or FRAMEWORK_NOISE.match(line):
                continue
            kept.append(line)
        text = _clean(kept)
        if len(text) < 80:  # a heading or page fragment alone retrieves nothing useful
            continue
        for piece in splitter.split_text(text):
            chunks.append({"text": f"{title}, page {page.number}\n{piece}",
                           "metadata": {**base, "section_type": "framework_text", "page": page.number}})

    return _number(chunks, doc_id)


def split_playbook(path: Path, entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Playbook -> chunks per subcategory and section (statement, About, Suggested Actions, Documentation)."""

    doc_id, title = entry["document_id"], entry["title"]
    framework = load_framework()
    base = {"document_id": doc_id, "doc_type": entry["doc_type"], "source": path.name, "title": title, "corpus": "external"}

    # Walk the pages, collecting lines per (subcategory, section) with the page each section starts on.
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for page in read_pdf_pages(path):
        for raw in page.text.splitlines():
            line = raw.strip()
            if PLAYBOOK_NOISE.match(line):
                continue
            heading = PLAYBOOK_HEADING.match(line)
            if heading:
                current = {"subcategory_id": f"{heading[1]} {heading[2]}", "section": "statement", "page": page.number, "lines": []}
                sections.append(current)
                continue
            if current is None:
                continue  # front matter before the first subcategory
            if line in PLAYBOOK_SECTIONS:
                if PLAYBOOK_SECTIONS[line] == current["section"]:
                    continue  # "AI Transparency Resources" continues the Documentation section
                current = {"subcategory_id": current["subcategory_id"], "section": PLAYBOOK_SECTIONS[line], "page": page.number, "lines": []}
                sections.append(current)
                continue
            current["lines"].append(line)

    chunks: list[dict[str, Any]] = []
    splitter = _splitter()
    for sec in sections:
        if sec["section"] == "references":
            continue
        text = _clean(sec["lines"])
        if not text:
            continue
        sub = framework.subcategory(sec["subcategory_id"])
        label = SECTION_LABELS[sec["section"]]
        meta = {**base, "section_type": sec["section"], "subcategory_id": sub["id"],
                "category_id": sub["category_id"], "function": sub["function"], "page": sec["page"]}
        for piece in splitter.split_text(text):
            chunks.append({"text": f"{title} — {sub['id']} — {label}\n{piece}", "metadata": meta})

    return _number(chunks, doc_id)


def _number(chunks: list[dict[str, Any]], doc_id: str) -> list[dict[str, Any]]:
    for i, chunk in enumerate(chunks):
        chunk["id"] = f"{doc_id}#{i}"
        chunk["metadata"]["chunk_index"] = i
    return chunks


SPLITTERS = {"framework": split_framework, "playbook": split_playbook}
