"""Typed request/response models. Bad input is rejected before any tokens are spent."""

from pydantic import BaseModel, Field


class Answer(BaseModel):
    """Structured model output — the shape every answer must have."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources_needed: bool
    citations: list[str]  # document_ids of the retrieved passages the answer used.
    refused: bool  # True when the retrieved context could not answer the question.


class AskRequest(BaseModel):
    question: str
    model: str | None = None  # Optional per-request model override.
    top_k: int = Field(default=5, ge=1, le=20)  # How many chunks to retrieve.


class AskResponse(BaseModel):
    answer: Answer
    tokens_used: int  # Chat tokens + question-embedding tokens.
    model: str
    latency_ms: int  # Retrieval + generation.
    cost_usd: float  # Chat cost + embedding cost.
    retrieved_chunk_ids: list[str]  # Every chunk put in the prompt, best match first.


# Pinecone metadata values must be a string, number, boolean, or list of strings (no null/nested objects).
MetadataValue = str | int | float | bool | list[str]


class IngestRequest(BaseModel):
    """One document to chunk, embed, and store. Defaults let /ingest return a clear 400 instead of a 422."""

    text: str = ""
    document_id: str = ""
    source: str | None = None  # e.g. the original filename; defaults to "api" if omitted.
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)  # Extra filterable fields.


class IngestResponse(BaseModel):
    document_id: str
    chunks_indexed: int
    status: str
