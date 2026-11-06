"""Week 1 live demo — five stages in one file, built up live in class.

Session 2 adds RAG: /ingest stores chunks in Pinecone, /ask retrieves the top-k chunks
and answers only from them (prompt in rag_prompt.py), /debug/retrieve shows retrieval alone.
"""

import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

# Load .env from this folder so the key is found regardless of shell working directory.
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)

from vector_store import (  # noqa: E402 — after load_dotenv so env vars are set
    EMBEDDING_MODEL,
    embedding_cost_usd,
    get_settings,
    ingest_document,
    pinecone_health,
    query_similar,
)
from rag_prompt import REFUSAL_ANSWER, build_grounded_messages, clean_citations  # noqa: E402

# Reuse one client so TLS handshakes are not repeated on every request.
app = FastAPI()
client = OpenAI()  # Reads OPENAI_API_KEY from the environment; never hardcode keys.

# Stage 4 default — strong general model; swap at request time for the live demo.
DEFAULT_MODEL = "gpt-4o"

# Stage 5 — per-1K-token input/output USD (derived from OpenAI list prices).
MODEL_PRICES_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "o3-mini": (0.0011, 0.0044),
}


class Answer(BaseModel):
    """Structured model output — this is what turns a chatbot into a component."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources_needed: bool
    citations: list[str]  # Session 2 — document_ids of the retrieved passages the answer used.
    refused: bool  # Session 2 — true when the retrieved context could not answer the question.


class AskRequest(BaseModel):
    """Typed request body so bad input is rejected before we spend tokens."""

    question: str
    force_bad: bool = False  # Stage 3 demo knob — first attempt breaks schema on purpose.
    model: str | None = None  # Stage 4 — optional override to swap models live.
    top_k: int = Field(default=5, ge=1, le=20)  # Session 2 — how many chunks to retrieve.


class AskResponse(BaseModel):
    """Typed response so callers always get the same shape back."""

    answer: Answer
    tokens_used: int  # Session 2: chat tokens + question-embedding tokens.
    model: str
    latency_ms: int  # Session 2: retrieval + generation.
    cost_usd: float  # Session 2: chat cost + embedding cost.
    retrieved_chunk_ids: list[str]  # Session 2 — every chunk put in the prompt, best match first.


def compute_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Turn real usage into dollars — same prompt, different model, different cost."""

    prices = MODEL_PRICES_PER_1K.get(model, MODEL_PRICES_PER_1K[DEFAULT_MODEL])
    input_per_1k, output_per_1k = prices
    return (prompt_tokens / 1000 * input_per_1k) + (completion_tokens / 1000 * output_per_1k)


def call_model_structured(messages: list[dict[str, str]], model: str) -> tuple[Answer, int, int, int]:
    """
    Stage 2 center: OpenAI structured output forces exactly the Answer schema.
    Session 2: takes the grounded messages (system rules + retrieved context + question).
    Returns parsed answer plus token counts from billing metadata.
    """

    completion = client.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=Answer,
    )

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Model returned no parseable structured output")

    usage = completion.usage
    total = usage.total_tokens if usage else 0
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    return parsed, total, prompt_tokens, completion_tokens


def call_model_unsafe(messages: list[dict[str, str]], model: str) -> tuple[Answer, int, int, int]:
    """
    Stage 3 demo path: free-form JSON call, then validate locally.
    The bad instruction makes confidence a string so Pydantic rejects it reliably.
    """

    *rest, last = messages
    bad_last = {
        **last,
        "content": (
            f"{last['content']}\n\n"
            "Reply with ONLY a JSON object using keys answer, confidence, sources_needed, citations, refused. "
            "Set confidence to the string 'very high' (not a number)."
        ),
    }
    completion = client.chat.completions.create(model=model, messages=[*rest, bad_last])

    raw = completion.choices[0].message.content or ""
    # Guardrail: refuse malformed output instead of passing it through to clients.
    answer = Answer.model_validate_json(raw)

    usage = completion.usage
    total = usage.total_tokens if usage else 0
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    return answer, total, prompt_tokens, completion_tokens


# Example:
#   curl -s -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
#     -d '{"question": "What is the mileage rate?"}'
@app.post("/ask")
def ask(body: AskRequest) -> AskResponse:
    """RAG answer: retrieve top-k chunks, answer only from them, with citations and cost visibility."""

    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="'question' must not be empty.")
    model = body.model or DEFAULT_MODEL
    start = time.perf_counter()

    # Session 2, steps 1-2: embed the question (same model as ingest) and retrieve top-k chunks.
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

    # Step 3: grounding prompt (see rag_prompt.py).
    messages = build_grounded_messages(question, matches)
    last_error: str | None = None

    # Stage 3: one retry keeps the logic legible while still protecting callers.
    for attempt in range(2):
        try:
            # Step 4: existing Session 1 generation path.
            # First attempt with force_bad uses the unsafe path; retry uses structured output.
            use_bad_path = body.force_bad and attempt == 0
            if use_bad_path:
                answer, tokens_used, prompt_tokens, completion_tokens = call_model_unsafe(messages, model)
            else:
                answer, tokens_used, prompt_tokens, completion_tokens = call_model_structured(messages, model)

            # Guardrail: citations must point at chunks we actually retrieved; refusals cite nothing.
            answer.citations = [] if answer.refused else clean_citations(answer.citations, matches)

            # Step 5: keep Session 1's cost visibility, now including the question embedding.
            latency_ms = int((time.perf_counter() - start) * 1000)
            cost_usd = compute_cost_usd(model, prompt_tokens, completion_tokens) + embed_cost

            return AskResponse(
                answer=answer,
                tokens_used=tokens_used + embed_tokens,
                model=model,
                latency_ms=latency_ms,
                cost_usd=round(cost_usd, 6),
                retrieved_chunk_ids=chunk_ids,  # Step 6.
            )
        except (ValidationError, ValueError) as exc:
            last_error = str(exc)
            continue

    # Clean failure — never leak a half-parsed response to the client.
    raise HTTPException(
        status_code=502,
        detail=f"Model response failed schema validation after retry: {last_error}",
    )


# Pinecone metadata values must be a string, number, boolean, or list of strings (no null/nested objects).
MetadataValue = str | int | float | bool | list[str]


class IngestRequest(BaseModel):
    """One document to chunk, embed, and store. Defaults let us return a clear 400 instead of a 422."""

    text: str = ""
    document_id: str = ""
    source: str | None = None  # e.g. the original filename; defaults to "api" if omitted.
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)  # Extra filterable fields.


class IngestResponse(BaseModel):
    document_id: str
    chunks_indexed: int
    status: str


# Example (bash / Git Bash / Render shell):
#   curl -s -X POST http://127.0.0.1:8000/ingest \
#     -H "Content-Type: application/json" \
#     -d '{"document_id": "refund-policy", "source": "refund_policy.md",
#          "text": "Refunds are available within 30 days of purchase...",
#          "metadata": {"team": "support"}}'
#   -> {"document_id":"refund-policy","chunks_indexed":1,"status":"indexed"}
#
# PowerShell (curl there is an alias, so use Invoke-RestMethod):
#   Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ingest -ContentType "application/json" `
#     -Body '{"document_id": "refund-policy", "source": "refund_policy.md", "text": "Refunds are available within 30 days..."}'
#
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


# Example:
#   curl -s "http://127.0.0.1:8000/debug/retrieve?q=how%20many%20days%20can%20I%20work%20remotely"
#   or just open that URL in a browser (spaces are fine there).
@app.get("/debug/retrieve")
def debug_retrieve(
    q: str = "",
    top_k: int = Query(default=5, ge=1, le=20),
) -> dict:
    """Retrieval only: embed q, return the closest chunks with scores. Never calls the LLM."""

    question = q.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required, e.g. /debug/retrieve?q=refund+policy")

    try:
        start = time.perf_counter()
        matches, tokens = query_similar(question, top_k=top_k)
        latency_ms = int((time.perf_counter() - start) * 1000)
        namespace = get_settings().namespace
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
