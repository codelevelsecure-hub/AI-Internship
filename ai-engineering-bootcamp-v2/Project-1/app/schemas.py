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


class AgentAnswer(ModelAnswer):
    """The agent's final structured answer (same fields as ModelAnswer, plus the live-search flag)."""

    needs_live_search: bool  # True when the question needs recent information beyond the documents.


class Answer(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    gap_noted: bool
    refused: bool
    standard_citations: list[Citation]
    org_citations: list[Citation]
    needs_live_search: bool = False  # Set by the agent when only live web search could answer (v2).


class ToolCallTrace(BaseModel):
    step: int
    tool: str
    arguments: dict
    summary: str  # Short description of what the tool returned.


class AskRequest(BaseModel):
    question: str
    model: str | None = None  # Optional per-request model override.
    top_k: int = Field(default=5, ge=1, le=20)  # Chunks to retrieve from EACH corpus.


class AskResponse(BaseModel):
    answer: Answer
    tokens_used: int  # Chat tokens + question-embedding tokens.
    model: str
    latency_ms: int  # Retrieval + generation.
    cost_usd: float  # Chat cost + embedding cost (+ gap-analysis cost when the agent ran it).
    retrieved_chunk_ids: list[str]  # Every chunk put in the prompt (standards first), best match first.
    mode: Literal["agent", "simple"] = "simple"
    tool_trace: list[ToolCallTrace] = Field(default_factory=list)  # Agent mode: the tools it used, in order.


# ---------------------------------------------------------------- gap analysis

GapStatus = Literal["Covered", "Partial", "Missing", "Not applicable"]
Priority = Literal["High", "Medium", "Low"]


class ModelRecommendation(BaseModel):
    action: str  # One concrete step, written for this organization.
    priority: Priority
    owner_hint: str  # Role that should own it, e.g. "AI Governance Committee".
    playbook_labels: list[str]  # P1, P2, ... — the Suggested Actions it draws on.


class GapAssessment(BaseModel):
    """What the model must return for one subcategory (structured output)."""

    status: GapStatus
    rationale: str  # Two to four sentences, citing E/P labels in brackets.
    evidence_labels: list[str]  # E1, E2, ... — organization passages that support the status.
    mapping_assessment: str  # Whether the mapped control's documented content fits the subcategory.
    recommendations: list[ModelRecommendation]  # Empty when Covered or Not applicable.
    confidence: float = Field(ge=0.0, le=1.0)


class EvidenceRef(BaseModel):
    label: str
    document_id: str
    title: str
    section: str | None = None
    date: str | None = None
    chunk_id: str
    quote: str


class PlaybookRef(BaseModel):
    label: str
    page: int | None = None
    chunk_id: str


class Recommendation(BaseModel):
    action: str
    priority: Priority
    owner_hint: str
    playbook_refs: list[PlaybookRef]


class GapFinding(BaseModel):
    subcategory_id: str
    function: str
    category_id: str
    statement: str
    page: int  # AI RMF 1.0 PDF page of the statement.
    mapped_controls: list[str]
    status: GapStatus
    rationale: str
    mapping_assessment: str
    evidence: list[EvidenceRef]
    recommendations: list[Recommendation]
    confidence: float
    cached: bool  # True when served from the cache (no tokens spent).
    cost_usd: float


class GapRequest(BaseModel):
    function: Literal["Govern", "Map", "Measure", "Manage"] | None = None  # None = all 72 subcategories.
    subcategory_ids: list[str] | None = None  # Or an explicit list (overrides function).
    model: str | None = None
    refresh: bool = False  # True ignores the cache and re-assesses.


class GapReport(BaseModel):
    generated_at: str
    model: str
    scope: str
    counts: dict[str, int]  # By status.
    counts_by_function: dict[str, dict[str, int]]
    findings: list[GapFinding]
    cached_findings: int
    cost_usd: float
    latency_ms: int


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
