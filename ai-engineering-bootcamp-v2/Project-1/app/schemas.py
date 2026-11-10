"""Typed request/response models. Bad input is rejected before any tokens are spent."""

from typing import Literal

from pydantic import BaseModel, Field


class ModelAnswer(BaseModel):
    """What the model must return (structured output). Citations are passage labels like 'S1' or 'O2'."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    cited_labels: list[str]
    gap_noted: bool  # True when NIST expects something the organization's documents don't show.
    refused: bool  # True only when neither corpus is relevant to the question.


class Citation(BaseModel):
    label: str  # The [S1]/[O2] marker used in the answer text.
    corpus: Literal["standard", "organization"]
    document_id: str
    title: str
    chunk_id: str
    subcategory_id: str | None = None
    page: int | None = None  # PDF page (standards only).
    section: str | None = None  # Heading in the organization document.
    quote: str  # Start of the cited passage, for display.


class Answer(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    gap_noted: bool
    refused: bool
    standard_citations: list[Citation]
    org_citations: list[Citation]


class AskRequest(BaseModel):
    question: str
    model: str | None = None  # Optional per-request model override.
    top_k: int = Field(default=5, ge=1, le=20)  # Chunks to retrieve from EACH corpus.


class AskResponse(BaseModel):
    answer: Answer
    tokens_used: int  # Chat tokens + question-embedding tokens.
    model: str
    latency_ms: int  # Retrieval + generation.
    cost_usd: float  # Chat cost + embedding cost.
    retrieved_chunk_ids: list[str]  # Every chunk put in the prompt (standards first), best match first.


# Pinecone metadata values must be a string, number, boolean, or list of strings (no null/nested objects).
MetadataValue = str | int | float | bool | list[str]


class IngestRequest(BaseModel):
    """One ad-hoc organization document. Defaults let /ingest return a clear 400 instead of a 422."""

    text: str = ""
    document_id: str = ""
    source: str | None = None  # e.g. the original filename; defaults to "api" if omitted.
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)  # Extra filterable fields.


class IngestResponse(BaseModel):
    document_id: str
    chunks_indexed: int
    status: str
