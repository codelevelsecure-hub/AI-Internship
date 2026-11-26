"""Ingest the corpus into Pinecone, straight from this machine (no API server needed).

    python ingest_docs.py --dry-run              # show files, namespaces and chunk counts; send nothing
    python ingest_docs.py                        # ingest both corpora
    python ingest_docs.py --corpus internal      # only the organization's documents

corpus/external/  -> standards namespace  (NIST PDFs, split by subcategory; see manifest.json)
corpus/internal/  -> organization namespace (text files, split by heading; see manifest.json)
The Control Mapping is also stored as one "crosswalk" record per row.

Only files listed in a folder's manifest.json are ingested. Re-running replaces each document's
chunks (no duplicates), so edit a file and just run it again. Laptop and deployed API read the
same Pinecone index, so one run serves both.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from app.ingestion.control_mapping import CROSSWALK_DOCUMENT_ID, crosswalk_chunks, parse_control_mapping
from app.ingestion.internal_splitter import split_internal
from app.ingestion.nist_splitter import SPLITTERS
from app.retrieval import embedding_cost_usd, get_settings, replace_document

ROOT = Path(__file__).resolve().parent
EXTERNAL, INTERNAL = ROOT / "corpus" / "external", ROOT / "corpus" / "internal"
SUPPORTED = {".pdf", ".txt", ".md"}


def load_manifest(folder: Path) -> dict[str, Any]:
    path = folder / "manifest.json"
    if not path.is_file():
        raise SystemExit(f"Missing {path} — list each file there with its document_id and doc_type.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    listed = set(manifest["documents"])
    present = {p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED}
    for name in sorted(present - listed):
        print(f"SKIP  {folder.name}/{name}: not listed in manifest.json")
    for name in sorted(listed - present):
        raise SystemExit(f"{folder.name}/manifest.json lists '{name}', but the file is missing.")
    return manifest


def plan_external() -> list[tuple[str, list[dict[str, Any]], str]]:
    manifest = load_manifest(EXTERNAL)
    plan = []
    for name, entry in manifest["documents"].items():
        splitter = SPLITTERS.get(entry["doc_type"])
        if splitter is None:
            raise SystemExit(f"external/{name}: no splitter for doc_type '{entry['doc_type']}'")
        plan.append((entry["document_id"], splitter(EXTERNAL / name, entry), name))
    return plan


def plan_internal() -> list[tuple[str, list[dict[str, Any]], str]]:
    manifest = load_manifest(INTERNAL)
    org = manifest["organization"]
    plan = []
    for name, entry in manifest["documents"].items():
        plan.append((entry["document_id"], split_internal(INTERNAL / name, entry, org), name))
        if entry["doc_type"] == "control_mapping":
            records = parse_control_mapping(INTERNAL / name)
            plan.append((CROSSWALK_DOCUMENT_ID, crosswalk_chunks(records, org, name), f"{name} (crosswalk records)"))
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest corpus/external and corpus/internal into Pinecone.")
    parser.add_argument("--corpus", choices=["all", "external", "internal"], default="all")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be ingested; send nothing")
    args = parser.parse_args()

    settings = get_settings()
    jobs = []
    if args.corpus in ("all", "external"):
        jobs += [(settings.standards_namespace, *item) for item in plan_external()]
    if args.corpus in ("all", "internal"):
        jobs += [(settings.org_namespace, *item) for item in plan_internal()]

    total_chunks = total_tokens = 0
    for namespace, document_id, chunks, label in jobs:
        if args.dry_run:
            chars = sum(len(c["text"]) for c in chunks)
            print(f"DRY   {label:<52} -> {namespace} / {document_id}: {len(chunks)} chunks, {chars:,} chars")
            total_chunks += len(chunks)
            continue
        try:
            indexed, tokens = replace_document(document_id, chunks, namespace)
        except Exception as exc:  # OpenAI or Pinecone failure: stop, nothing half-explained
            print(f"FAIL  {label}: {type(exc).__name__}: {exc}")
            return 1
        total_chunks += indexed
        total_tokens += tokens
        print(f"OK    {label:<52} -> {namespace} / {document_id}: {indexed} chunks")

    if args.dry_run:
        print(f"\n{total_chunks} chunks would be ingested (nothing sent).")
    else:
        print(f"\n{total_chunks} chunks ingested, {total_tokens:,} embedding tokens, ${embedding_cost_usd(total_tokens):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
