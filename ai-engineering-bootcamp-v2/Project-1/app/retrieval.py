"""Pinecone vector store + OpenAI embeddings.

All config comes from environment variables (see .env.example), loaded by the
app package. Nothing here talks to Pinecone or OpenAI at import time, so the app
still starts if Pinecone isn't configured yet; /health/pinecone tells you what's
missing.

Quick check from the Project-1 folder:  python -m app.retrieval
"""

import os
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone

# Fixed on purpose (not an env var): ingest and query MUST use the same model,
# or query vectors land in a different space than the stored ones.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # Your Pinecone index must be created with this dimension.
EMBEDDING_PRICE_PER_1K = 0.00002  # USD, text-embedding-3-small list price ($0.02 / 1M tokens).

# Chunking defaults (characters, not tokens). Override with CHUNK_SIZE / CHUNK_OVERLAP.
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100

BATCH_SIZE = 100  # Chunks per OpenAI embeddings call and per Pinecone upsert request.
DELETE_BATCH_SIZE = 1000  # Pinecone's max ids per delete call.


@dataclass(frozen=True)
class PineconeSettings:
    api_key: str
    index_name: str
    namespace: str


def get_settings() -> PineconeSettings:
    """Read Pinecone config from the environment; fail loudly if a required var is missing."""

    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    index_name = os.getenv("PINECONE_INDEX_NAME", "").strip()
    missing = [name for name, value in (("PINECONE_API_KEY", api_key), ("PINECONE_INDEX_NAME", index_name)) if not value]
    if missing:
        raise RuntimeError(f"Missing required env var(s): {', '.join(missing)}")
    namespace = os.getenv("PINECONE_NAMESPACE", "default").strip() or "default"
    return PineconeSettings(api_key=api_key, index_name=index_name, namespace=namespace)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from None


def chunk_settings() -> tuple[int, int]:
    """(chunk_size, chunk_overlap) from env, validated so the splitter can't be misconfigured."""

    size = _env_int("CHUNK_SIZE", DEFAULT_CHUNK_SIZE)
    overlap = _env_int("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)
    if size <= 0 or overlap < 0 or overlap >= size:
        raise RuntimeError(f"Need CHUNK_SIZE > 0 and 0 <= CHUNK_OVERLAP < CHUNK_SIZE (got {size}, {overlap})")
    return size, overlap


@lru_cache(maxsize=1)
def _openai() -> OpenAI:
    return OpenAI()  # Reads OPENAI_API_KEY from the environment.


@lru_cache(maxsize=1)
def get_index():
    """One cached Pinecone Index handle per process (avoids re-resolving the host each request)."""

    settings = get_settings()
    return Pinecone(api_key=settings.api_key).Index(name=settings.index_name)


def embedding_cost_usd(tokens: int) -> float:
    return tokens / 1000 * EMBEDDING_PRICE_PER_1K


def split_text(text: str) -> list[str]:
    """Split on paragraphs, then lines, then words, so chunks break at natural boundaries."""

    size, overlap = chunk_settings()
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return [chunk for chunk in splitter.split_text(text) if chunk.strip()]


def embed_texts(texts: list[str]) -> tuple[list[list[float]], int]:
    """Embed texts in batches. Returns (vectors in input order, total tokens billed)."""

    vectors: list[list[float]] = []
    tokens = 0
    for start in range(0, len(texts), BATCH_SIZE):
        response = _openai().embeddings.create(model=EMBEDDING_MODEL, input=texts[start : start + BATCH_SIZE])
        vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        tokens += response.usage.total_tokens
    return vectors, tokens


def upsert_chunks(chunks: list[dict[str, Any]]) -> tuple[int, int]:
    """
    Embed and store chunks. Each chunk: {"id": str, "text": str, "metadata": dict (optional)}.
    Stable ids mean re-ingesting the same chunk overwrites it instead of duplicating it.
    Returns (vectors upserted, embedding tokens used).
    """

    if not chunks:
        return 0, 0
    vectors, tokens = embed_texts([c["text"] for c in chunks])
    records = [
        {
            "id": c["id"],
            "values": vector,
            # Keep the text in metadata so a query returns something you can put in a prompt.
            "metadata": {**c.get("metadata", {}), "text": c["text"]},
        }
        for c, vector in zip(chunks, vectors)
    ]
    response = get_index().upsert(
        vectors=records, namespace=get_settings().namespace, batch_size=BATCH_SIZE, show_progress=False
    )
    return response.upserted_count, tokens


def _chunk_id_prefix(document_id: str) -> str:
    return f"{document_id}#"


def _delete_stale_chunks(document_id: str, keep_ids: set[str]) -> int:
    """Remove chunks left over from a longer, earlier version of this document."""

    index = get_index()
    namespace = get_settings().namespace
    stale = [
        item.id
        for page in index.list(prefix=_chunk_id_prefix(document_id), namespace=namespace)
        for item in page.vectors
        if item.id not in keep_ids
    ]
    for start in range(0, len(stale), DELETE_BATCH_SIZE):
        index.delete(ids=stale[start : start + DELETE_BATCH_SIZE], namespace=namespace)
    return len(stale)


def ingest_document(
    document_id: str, text: str, source: str, metadata: dict[str, Any] | None = None
) -> tuple[int, int]:
    """
    Chunk -> embed -> upsert one document. Re-ingesting the same document_id replaces it:
    chunk ids are "<document_id>#<chunk_index>", so new chunks overwrite old ones and any
    extra old chunks are deleted afterwards (upsert first, so the doc is never missing).
    Returns (chunks indexed, embedding tokens used).
    """

    pieces = split_text(text)
    chunks = [
        {
            "id": f"{_chunk_id_prefix(document_id)}{i}",
            "text": piece,
            # Caller metadata first, so the required fields can't be overwritten by it.
            "metadata": {**(metadata or {}), "document_id": document_id, "chunk_index": i, "source": source},
        }
        for i, piece in enumerate(pieces)
    ]
    indexed, tokens = upsert_chunks(chunks)
    _delete_stale_chunks(document_id, keep_ids={c["id"] for c in chunks})
    return indexed, tokens


def query_similar(text: str, top_k: int = 4) -> tuple[list[dict[str, Any]], int]:
    """Embed the query with the SAME model and return the top_k closest chunks plus tokens used."""

    vectors, tokens = embed_texts([text])
    result = get_index().query(
        vector=vectors[0],
        top_k=top_k,
        namespace=get_settings().namespace,
        include_metadata=True,
    )
    matches = [
        {"id": m.id, "score": m.score, "text": (m.metadata or {}).get("text", ""), "metadata": m.metadata or {}}
        for m in result.matches
    ]
    return matches, tokens


def pinecone_health() -> dict[str, Any]:
    """
    Debug check: is Pinecone reachable and is the index shaped right for our embeddings?
    Makes one cheap describe_index_stats call. Costs no OpenAI tokens and never raises.
    """

    report: dict[str, Any] = {"ok": False, "embedding_model": EMBEDDING_MODEL, "expected_dimension": EMBEDDING_DIM}
    try:
        settings = get_settings()
        report.update(index=settings.index_name, namespace=settings.namespace)

        start = time.perf_counter()
        stats = get_index().describe_index_stats()
        report["latency_ms"] = int((time.perf_counter() - start) * 1000)

        ns = (stats.namespaces or {}).get(settings.namespace)
        report.update(
            index_dimension=stats.dimension,
            metric=stats.metric,
            total_vector_count=stats.total_vector_count,
            namespace_vector_count=ns.vector_count if ns else 0,
        )
        if stats.dimension != EMBEDDING_DIM:
            report["error"] = (
                f"Index dimension is {stats.dimension} but {EMBEDDING_MODEL} produces {EMBEDDING_DIM}. "
                "Delete it and recreate with create_pinecone_index.py (one dense field, 1536, cosine)."
            )
            return report
        report["ok"] = True
    except Exception as exc:  # Debug endpoint: report the problem instead of crashing.
        get_index.cache_clear()  # Don't keep a broken handle around after fixing config.
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


if __name__ == "__main__":
    import json

    print(json.dumps(pinecone_health(), indent=2, default=str))
