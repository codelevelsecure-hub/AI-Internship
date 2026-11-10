"""HTTP routes for AI Trust Navigator.

/ask answers compliance questions from the NIST AI RMF and the organization's documents,
/debug/retrieve shows retrieval alone, /ingest adds an ad-hoc organization document.
"""

import time

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from app import auth, crosswalk, gap_analysis, qa
from app import evidence as find_evidence
from app.agents import router
from app.frameworks import load_framework
from app.retrieval import EMBEDDING_MODEL, embedding_cost_usd, ingest_document, pinecone_health
from app.schemas import AskRequest, AskResponse, GapReport, GapRequest, IngestRequest, IngestResponse
from app.search import search_both

app = FastAPI(title="AI Trust Navigator")
app.middleware("http")(auth.require_api_key)  # X-API-Key on every route except health and docs


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
    """Agent mode: the router picks tools (search, mapping, evidence, gap analysis, web search), then
    answers only from what they returned, citing NIST and organization passages separately."""

    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="'question' must not be empty.")
    try:
        return router.run_agent(question, model=body.model)
    except qa.AnswerFailed as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise _service_error(exc, "Answer") from exc


@app.post("/ask/simple")
def ask_simple(body: AskRequest) -> AskResponse:
    """Single-pass retrieval from both corpora (no tools): cheaper and faster, useful for comparison."""

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


@app.get("/frameworks/nist-ai-rmf/subcategories")
def list_subcategories(function: str | None = None) -> dict:
    """The 72 NIST AI RMF 1.0 subcategories (optionally one function), each with its mapped controls."""

    fw = load_framework()
    if function and function.title() not in fw.functions:
        raise HTTPException(status_code=400, detail=f"'function' must be one of {fw.functions}")
    try:
        mapped = {r["subcategory_id"]: [] for r in crosswalk.load_records()}
        for r in crosswalk.load_records():
            mapped[r["subcategory_id"]].append(r["control_name"])
    except Exception as exc:
        raise _service_error(exc, "Loading the control mapping") from exc
    subs = [s for s in fw.subcategories if not function or s["function"] == function.title()]
    return {
        "framework": fw.name,
        "version": fw.version,
        "count": len(subs),
        "subcategories": [{**s, "mapped_controls": mapped.get(s["id"], [])} for s in subs],
    }


@app.get("/controls")
def list_controls() -> dict:
    """The organization's controls from its control mapping, with the subcategories each one covers."""

    try:
        items = crosswalk.controls()
    except Exception as exc:
        raise _service_error(exc, "Loading the control mapping") from exc
    return {"count": len(items), "controls": items}


# Example: http://127.0.0.1:8001/evidence?subcategory=MANAGE%204.3  or  ?control=AI%20Incident%20Response
@app.get("/evidence")
def evidence(subcategory: str | None = None, control: str | None = None, top_k: int = Query(default=5, ge=1, le=10)) -> dict:
    """Organization evidence (chunk-level quotes) for one NIST subcategory or one control. No LLM call."""

    if bool(subcategory) == bool(control):
        raise HTTPException(status_code=400, detail="Pass exactly one of 'subcategory' or 'control'.")
    try:
        if subcategory:
            return find_evidence.evidence_for_subcategory(subcategory, top_k=top_k)
        return find_evidence.evidence_for_control(control, top_k=top_k)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc
    except Exception as exc:
        raise _service_error(exc, "Evidence lookup") from exc


# Example (PowerShell): Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8001/gaps -ContentType "application/json" -Body '{"function": "Govern"}'
@app.post("/gaps")
def gaps(body: GapRequest | None = Body(default=None)) -> GapReport:
    """Assess subcategories (all 72, one function, or a list) against the organization's documents.

    Unchanged subcategories are served from the cache at no token cost; refresh=true re-assesses them.
    """

    body = body or GapRequest()
    try:
        return gap_analysis.run_gap_analysis(function=body.function, subcategory_ids=body.subcategory_ids,
                                             model=body.model, refresh=body.refresh)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc.args[0])) from exc
    except Exception as exc:
        raise _service_error(exc, "Gap analysis") from exc


@app.get("/gaps/latest")
def gaps_latest() -> GapReport:
    """The most recent gap report (no new model calls). 404 until a report has been run."""

    report = gap_analysis.latest_report()
    if report is None:
        raise HTTPException(status_code=404, detail="No gap report yet. Run POST /gaps first.")
    return report


@app.get("/health/pinecone")
def health_pinecone() -> JSONResponse:
    """Debug check: Pinecone reachable, index dimension matches, vectors in both namespaces. 503 if not."""

    report = {**pinecone_health(), "auth_enabled": auth.auth_enabled()}
    return JSONResponse(report, status_code=200 if report["ok"] else 503)
