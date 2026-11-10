"""HTTP routes for AI Trust Navigator.

/ask answers compliance questions from the NIST AI RMF and the organization's documents,
/debug/retrieve shows retrieval alone, /ingest adds an ad-hoc organization document.
"""

import time

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from app import qa
from app.retrieval import EMBEDDING_MODEL, embedding_cost_usd, ingest_document, pinecone_health
from app.schemas import AskRequest, AskResponse, IngestRequest, IngestResponse
from app.search import search_both

app = FastAPI(title="AI Trust Navigator")


def _service_error(exc: Exception, action: str) -> HTTPException:
    """Missing configuration -> 503 (not the caller's fault); OpenAI/Pinecone failure -> 502."""

    if isinstance(exc, RuntimeError):
        return HTTPException(status_code=503, detail=f"Service not configured: {exc}")
    return HTTPException(status_code=502, detail=f"{action} failed: {type(exc).__name__}: {exc}")


# Example (PowerShell):
#   Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8001/ask -ContentType "application/json" `
#     -Body '{"question": "Who approves high-risk AI systems?"}'
@app.post("/ask")
def ask(body: AskRequest) -> AskResponse:
    """Retrieve from both corpora, answer only from them, cite standard and organization passages separately."""

    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="'question' must not be empty.")
    try:
        return qa.answer_question(question, model=body.model, top_k=body.top_k)
    except qa.AnswerFailed as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise _service_error(exc, "Answer") from exc


# Re-posting the same document_id replaces that document's chunks (no duplicates).
@app.post("/ingest")
def ingest(body: IngestRequest | None = Body(default=None)) -> IngestResponse:
    """Chunk an ad-hoc organization document, embed it, and store it in the organization namespace."""

    # Guardrail: reject empty input before spending any tokens.
    if body is None:
        raise HTTPException(status_code=400, detail="Request body is required: {\"text\": ..., \"document_id\": ...}")
    document_id = body.document_id.strip()
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="'text' is required and must not be empty or whitespace.")
    if not document_id:
        raise HTTPException(status_code=400, detail="'document_id' is required and must not be empty.")
    if not document_id.isascii() or len(document_id) > 200:
        raise HTTPException(status_code=400, detail="'document_id' must be ASCII and at most 200 characters.")

    try:
        chunks_indexed, _tokens = ingest_document(
            document_id=document_id,
            text=body.text,
            source=(body.source or "").strip() or "api",
            metadata=body.metadata,
        )
    except Exception as exc:
        raise _service_error(exc, "Ingest") from exc
    return IngestResponse(document_id=document_id, chunks_indexed=chunks_indexed, status="indexed")


# Example: open http://127.0.0.1:8001/debug/retrieve?q=incident%20response in a browser.
@app.get("/debug/retrieve")
def debug_retrieve(q: str = "", top_k: int = Query(default=5, ge=1, le=20)) -> dict:
    """Retrieval only, from both corpora: labelled passages with scores. Never calls the LLM."""

    question = q.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required, e.g. /debug/retrieve?q=risk+assessment")
    try:
        start = time.perf_counter()
        retrieved = search_both(question, top_k=top_k)
        latency_ms = int((time.perf_counter() - start) * 1000)
    except Exception as exc:
        raise _service_error(exc, "Retrieve") from exc

    def row(m: dict) -> dict:
        meta = m["metadata"]
        return {
            "label": m["label"],
            "score": round(m["score"], 4),  # Cosine similarity: higher = closer.
            "document_id": meta.get("document_id"),
            "subcategory_id": meta.get("subcategory_id"),
            "section_type": meta.get("section_type"),
            "section": meta.get("section"),
            "page": meta.get("page"),
            "chunk_id": m["id"],
            "text": m["text"],
        }

    return {
        "query": question,
        "embedding_model": EMBEDDING_MODEL,
        "named_subcategories": retrieved.named_subcategories,
        "top_k": top_k,
        "latency_ms": latency_ms,
        "tokens_used": retrieved.embedding_tokens,  # Embedding tokens only — no LLM call happens here.
        "cost_usd": round(embedding_cost_usd(retrieved.embedding_tokens), 8),
        "standards": [row(m) for m in retrieved.standards],
        "organization": [row(m) for m in retrieved.organization],
    }


@app.get("/health/pinecone")
def health_pinecone() -> JSONResponse:
    """Debug check: Pinecone reachable, index dimension matches, vectors in both namespaces. 503 if not."""

    report = pinecone_health()
    return JSONResponse(report, status_code=200 if report["ok"] else 503)
