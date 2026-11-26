"""Answer a compliance question from both corpora, with label-based citations and cost visibility."""

import re
import time
from typing import Any

from pydantic import ValidationError

from app import crosswalk
from app.llm import DEFAULT_MODEL, call_model_structured, compute_cost_usd
from app.prompts import REFUSAL_ANSWER, build_messages, clean_labels
from app.retrieval import embedding_cost_usd
from app.schemas import Answer, AskResponse, Citation, ModelAnswer
from app.search import search_both

INLINE_LABELS = re.compile(r"\[([SO]\d+(?:\s*[,;]\s*[SO]\d+)*)\]")
QUOTE_CHARS = 240


class AnswerFailed(Exception):
    """The model's output failed schema validation twice."""


def _as_int(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


def to_citation(match: dict[str, Any]) -> Citation:
    meta = match["metadata"]
    body = match["text"].split("\n", 1)[-1].strip()  # drop the "<title> — <section>" header line
    return Citation(
        label=match["label"],
        corpus=match["corpus"],
        document_id=meta.get("document_id", "unknown"),
        title=meta.get("title") or meta.get("source", "unknown"),
        chunk_id=match["id"],
        subcategory_id=meta.get("subcategory_id"),
        page=_as_int(meta.get("page")),
        section=meta.get("section"),
        quote=body[:QUOTE_CHARS] + ("…" if len(body) > QUOTE_CHARS else ""),
    )


def answer_question(question: str, model: str | None = None, top_k: int = 5) -> AskResponse:
    """Raises RuntimeError for missing configuration, AnswerFailed after two bad model outputs."""

    model = model or DEFAULT_MODEL
    start = time.perf_counter()

    # Retrieval happens once, outside the retry loop, so a schema retry doesn't pay for it again.
    retrieved = search_both(question, top_k=top_k)
    embed_cost = embedding_cost_usd(retrieved.embedding_tokens)
    chunk_ids = [m["id"] for m in retrieved.all()]

    if not retrieved.all():  # Nothing indexed at all: refuse without paying for an LLM call.
        return AskResponse(
            answer=Answer(answer=REFUSAL_ANSWER, confidence=0.0, gap_noted=False, refused=True,
                          standard_citations=[], org_citations=[]),
            tokens_used=retrieved.embedding_tokens,
            model=model,
            latency_ms=int((time.perf_counter() - start) * 1000),
            cost_usd=round(embed_cost, 6),
            retrieved_chunk_ids=[],
        )

    mapping_lines = crosswalk.mapping_summary(retrieved.named_subcategories) if retrieved.named_subcategories else None
    messages = build_messages(question, retrieved.standards, retrieved.organization, mapping_lines)
    by_label = retrieved.by_label()

    last_error: str | None = None
    for _attempt in range(2):  # Guardrail: one retry if the output fails schema validation.
        try:
            parsed, tokens, prompt_tokens, completion_tokens = call_model_structured(messages, model, ModelAnswer)
        except (ValidationError, ValueError) as exc:
            last_error = str(exc)
            continue

        if parsed.refused:
            labels: list[str] = []
        else:
            # Labels named in the answer text count too, so a citation the model forgot to list isn't lost.
            inline = [group for group in INLINE_LABELS.findall(parsed.answer)]
            labels = clean_labels(parsed.cited_labels + inline, set(by_label))
        citations = [to_citation(by_label[label]) for label in labels]

        answer = Answer(
            answer=parsed.answer,
            confidence=parsed.confidence,
            gap_noted=parsed.gap_noted and not parsed.refused,
            refused=parsed.refused,
            standard_citations=[c for c in citations if c.corpus == "standard"],
            org_citations=[c for c in citations if c.corpus == "organization"],
        )
        return AskResponse(
            answer=answer,
            tokens_used=tokens + retrieved.embedding_tokens,
            model=model,
            latency_ms=int((time.perf_counter() - start) * 1000),
            cost_usd=round(compute_cost_usd(model, prompt_tokens, completion_tokens) + embed_cost, 6),
            retrieved_chunk_ids=chunk_ids,
        )

    raise AnswerFailed(f"Model response failed schema validation after retry: {last_error}")
