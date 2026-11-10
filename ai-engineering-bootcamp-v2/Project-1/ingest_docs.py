"""Send every .md / .txt file in a folder to POST /ingest.

Usage (from the week-1 folder, with the API running):
    python ingest_docs.py                      # docs/ -> http://127.0.0.1:8000
    python ingest_docs.py --dry-run            # show what would be sent, send nothing
    python ingest_docs.py --url https://<your-app>.onrender.com
    python ingest_docs.py path\\to\\other_folder

Each file becomes one document:
    document_id = file name without extension, lowercased, e.g. "Employee Handbook.md" -> "employee-handbook"
    source      = the file name, e.g. "Employee Handbook.md"
Re-running replaces those documents in Pinecone (no duplicates), so edit a file and just run it again.
"""

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import httpx

SUPPORTED = {".md", ".txt"}
DEFAULT_DIR = Path(__file__).resolve().parent / "docs"


def to_document_id(path: Path) -> str:
    """ASCII, lowercase, hyphenated — what /ingest accepts as a stable id."""

    ascii_name = unicodedata.normalize("NFKD", path.stem).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9._-]+", "-", ascii_name.lower()).strip("-")[:200] or "document"


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest a folder of .md/.txt files via POST /ingest.")
    parser.add_argument("folder", nargs="?", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--dry-run", action="store_true", help="List files and ids without sending")
    args = parser.parse_args()

    if not args.folder.is_dir():
        print(f"Folder not found: {args.folder}  (create it and put .md/.txt files inside)")
        return 1
    files = sorted(p for p in args.folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED)
    if not files:
        print(f"No .md or .txt files in {args.folder}")
        return 1

    failures = 0
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        doc_id = to_document_id(path)
        if not text.strip():
            print(f"SKIP  {path.name}: file is empty")
            continue
        if args.dry_run:
            print(f"DRY   {path.name} -> document_id={doc_id!r} ({len(text):,} chars)")
            continue
        try:
            r = httpx.post(
                f"{args.url.rstrip('/')}/ingest",
                json={"document_id": doc_id, "source": path.name, "text": text},
                timeout=300.0,  # Large files mean many embedding calls.
            )
        except httpx.HTTPError as exc:
            print(f"FAIL  {path.name}: could not reach {args.url} ({exc}). Is uvicorn running?")
            return 1
        if r.status_code == 200:
            print(f"OK    {path.name} -> {doc_id!r}: {r.json()['chunks_indexed']} chunks")
        else:
            failures += 1
            print(f"FAIL  {path.name}: HTTP {r.status_code} {r.text[:300]}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
