"""Read a PDF page by page, keeping page numbers for citations."""

from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass(frozen=True)
class Page:
    number: int  # 1-based physical page number, as a PDF viewer shows it.
    text: str


def read_pdf_pages(path: Path) -> list[Page]:
    """Return every page's plain text (empty pages included, so numbering stays aligned)."""

    with pymupdf.open(path) as doc:
        return [Page(number=i, text=page.get_text("text")) for i, page in enumerate(doc, start=1)]
