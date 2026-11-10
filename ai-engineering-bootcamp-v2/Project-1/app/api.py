"""HTTP routes for AI Trust Navigator.

/ingest stores chunks in Pinecone, /ask retrieves the top-k chunks and answers only
from them (prompt in app/prompts.py), /debug/retrieve shows retrieval alone.
"""

import time

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.llm import DEFAULT_MODEL, call_model_structured, compute_cost_usd
from app.prompts import REFUSAL_ANSWER, build_grounded_messages, clean_citations
from app.retrieval import (
    EMBEDDING_MODEL,
    embedding_cost_usd,
    get_settings,
    ingest_document,
    pinecone_health,
    query_similar,
)
from app.schemas import Answer, AskRequest, AskResponse, IngestRequest, IngestResponse

app = FastAPI(title="AI Trust Navigator")


# Example:
#   curl -s -X POST http://127.0.0.1:8001/ask -H "Content-Type: application/json" \
#     -d '{"question": "Who approves high-risk AI systems?"}'
@app.post("/ask")
def ask(body: AskRequest) -> AskResponse:
    """RAG answer: retrieve top-k chunks, answer only from them, with citations and cost visibility."""

    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="'question' must not be empty.")
    model = body.model or DEFAULT_MODEL
    start = time.perf_counter()

    # Embed the question (same model as ingest) and retrieve the top-k chunks.
    # Done once, outside the retry loop, so a schema retry doesn't pay for retrieval again.
    try:
        matches, embed_tokens = query_similar(question, top_k=body.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f"Vector store not configured: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Retrieval failed: {type(exc).__name__}: {exc}") from exc
    chunk_ids = [m["id"] for m in matches]
    embed_cost = embedding_cost_usd(embed_tokens)

    # Nothing indexed at all: refuse without paying for an LLM call.
    if not matches:
        return AskResponse(
            answer=Answer(answer=REFUSAL_ANSWER, confidence=0.0, sources_needed=True, citations=[], refused=True),
            tokens_used=embed_tokens,
            model=model,
            latency_ms=int((time.perf_counter() - start) * 1000),
            cost_usd=round(embed_cost, 6),
            retrieved_chunk_ids=[],
        )

    messages = build_grounded_messages(question, matches)
    last_error: str | None = None

    # Guardrail: one retry if the model's output fails schema validation.
    for _attempt in range(2):
        try:
            answer, tokens_used, prompt_tokens, completion_tokens = call_model_structured(messages, model)

            # Citations must point at chunks we actually retrieved; refusals cite nothing.
            answer.citations = [] if answer.refused else clean_citations(answer.citations, matches)

            return AskResponse(
                answer=answer,
                tokens_used=tokens_used + embed_tokens,
                model=model,
                latency_ms=int((time.perf_counter() - start) * 1000),
                cost_usd=round(compute_cost_usd(model, prompt_tokens, completion_tokens) + embed_cost, 6),
                retrieved_chunk_ids=chunk_ids,
            )
        except (ValidationError, ValueError) as exc:
            last_error = str(exc)

    # Clean failure — never leak a half-parsed response to the client.
    raise HTTPException(
        status_code=502,
        detail=f"Model response failed schema validation after retry: {last_error}",
    )


# Example (PowerShell):
#   Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8001/ingest -ContentType "application/json" `
#     -Body '{"document_id": "ai-governance-policy", "source": "AI Governance Policy.txt", "text": "..."}'
# Re-posting the same document_id replaces that document's chunks (no duplicates).
@app.post("/ingest")
def ingest(body: IngestRequest | None = Body(default=None)) -> IngestResponse:
    """Chunk a document, embed each chunk with text-embedding-3-small, and upsert into Pinecone."""

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
    except RuntimeError as exc:  # Missing/invalid server config (env vars) — not the caller's fault.
        raise HTTPException(status_code=503, detail=f"Vector store not configured: {exc}") from exc
    except Exception as exc:  # OpenAI or Pinecone call failed.
        raise HTTPException(status_code=502, detail=f"Ingest failed: {type(exc).__name__}: {exc}") from exc

    return IngestResponse(document_id=document_id, chunks_indexed=chunks_indexed, status="indexed")


# Example: open http://127.0.0.1:8001/debug/retrieve?q=incident%20response in a browser.
@app.get("/debug/retrieve")
def debug_retrieve(
    q: str = "",
    top_k: int = Query(default=5, ge=1, le=20),
) -> dict:
    """Retrieval only: embed q, return the closest chunks with scores. Never calls the LLM."""

    question = q.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required, e.g. /debug/retrieve?q=risk+assessment")

    try:
        start = time.perf_counter()
        matches, tokens = query_similar(question, top_k=top_k)
        latency_ms = int((time.perf_counter() - start) * 1000)
        namespace = get_settings().org_namespace
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f"Vector store not configured: {exc}") from exc
    except Exception as exc:  # OpenAI embedding or Pinecone query failed.
        raise HTTPException(status_code=502, detail=f"Retrieve failed: {type(exc).__name__}: {exc}") from exc

    return {
        "query": question,
        "embedding_model": EMBEDDING_MODEL,
        "namespace": namespace,
        "top_k": top_k,
        "latency_ms": latency_ms,
        "tokens_used": tokens,  # Embedding tokens only — no LLM call happens here.
        "cost_usd": round(embedding_cost_usd(tokens), 8),
        "matches": [
            {
                "rank": rank,
                "score": round(m["score"], 4),  # Cosine similarity: higher = closer.
                "document_id": m["metadata"].get("document_id"),
                "chunk_index": m["metadata"].get("chunk_index"),
                "source": m["metadata"].get("source"),
                "chunk_id": m["id"],
                "text": m["text"],
            }
            for rank, m in enumerate(matches, start=1)
        ],
    }


@app.get("/health/pinecone")
def health_pinecone() -> JSONResponse:
    """Debug check: Pinecone reachable + index dimension matches our embedding model. 503 if not."""

    report = pinecone_health()
    return JSONResponse(report, status_code=200 if report["ok"] else 503)
