"""Split the organization's text documents on their "## " headings.

Each chunk is one section, prefixed with organization, document title and section heading so it
reads on its own in a prompt. Very short sections are merged into the next one, and long ones are
split further. In the system inventory, each "## System N" section is tagged with its system name.
"""

import re
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE, CHUNK_OVERLAP = 1200, 150
MIN_SECTION_CHARS = 150  # shorter sections are merged into the following one


def _sections(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (document title, [(heading, body), ...]). Text before the first heading is 'Overview'."""

    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    title = lines.pop(0).lstrip("# ").strip() if lines else ""

    sections: list[tuple[str, list[str]]] = [("Overview", [])]
    for line in lines:
        if line.startswith("## "):
            sections.append((line[3:].strip(), []))
        elif line.strip() != "---":  # horizontal rules carry no content
            sections[-1][1].append(line)

    cleaned = [(heading, re.sub(r"\n{3,}", "\n\n", "\n".join(body)).strip()) for heading, body in sections]
    return title, [(heading, body) for heading, body in cleaned if body]


def _system_name(body: str) -> str | None:
    match = re.search(r"^### (.+)$", body, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def split_internal(path: Path, entry: dict[str, Any], organization: str) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if entry.get("stop_at") and entry["stop_at"] in text:
        text = text[: text.index(entry["stop_at"])]

    title, sections = _sections(text)
    doc_id = entry["document_id"]
    base: dict[str, Any] = {
        "document_id": doc_id,
        "doc_type": entry["doc_type"],
        "source": path.name,
        "title": title,
        "corpus": "internal",
        "organization": organization,
    }
    for optional in ("date", "ai_system"):
        if entry.get(optional):
            base[optional] = entry[optional]

    # Merge short sections forward so no chunk is just a heading and a word.
    merged: list[tuple[str, str]] = []
    carry_heading, carry_body = None, ""
    for index, (heading, body) in enumerate(sections):
        # Label each section's body with its own heading once it shares a chunk with others.
        labelled = body if heading == "Overview" else f"{heading}:\n{body}"
        if carry_heading is not None:
            heading, labelled = f"{carry_heading} / {heading}", f"{carry_body}\n\n{labelled}"
            body = labelled
            carry_heading, carry_body = None, ""
        if len(body) < MIN_SECTION_CHARS and index < len(sections) - 1:
            carry_heading, carry_body = heading, labelled
            continue
        merged.append((heading, body))
    if carry_heading is not None:
        merged.append((carry_heading, carry_body))

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks: list[dict[str, Any]] = []
    for heading, body in merged:
        meta = {**base, "section": heading}
        system = _system_name(body)
        if system and entry["doc_type"] == "system_inventory":
            meta["ai_system"] = system
        for piece in splitter.split_text(body):
            chunks.append({"text": f"{organization} — {title} — {heading}\n{piece}", "metadata": meta})

    for i, chunk in enumerate(chunks):
        chunk["id"] = f"{doc_id}#{i}"
        chunk["metadata"] = {**chunk["metadata"], "chunk_index": i}
    return chunks
