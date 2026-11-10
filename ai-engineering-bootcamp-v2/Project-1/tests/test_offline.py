"""Offline tests: no OpenAI or Pinecone calls. Run from the Project-1 folder with `pytest`."""

import pytest
from fastapi.testclient import TestClient

import app.api as api
from app.prompts import clean_citations
from app.schemas import Answer

client = TestClient(api.app)


def _match(doc_id: str, idx: int) -> dict:
    return {"id": f"{doc_id}#{idx}", "score": 0.9, "text": "...", "metadata": {"document_id": doc_id, "source": f"{doc_id}.txt"}}


@pytest.fixture
def no_network(monkeypatch):
    """Fail loudly if a test reaches retrieval or the LLM without mocking them."""

    def boom(*_args, **_kwargs):
        raise AssertionError("network call attempted in an offline test")

    monkeypatch.setattr(api, "query_similar", boom)
    monkeypatch.setattr(api, "call_model_structured", boom)


def test_empty_question_is_rejected_before_any_call(no_network):
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400
    assert "must not be empty" in response.json()["detail"]


def test_ask_response_shape_and_citation_guardrail(monkeypatch):
    matches = [_match("ai-governance-policy", 0), _match("ai-risk-assessment", 2)]
    monkeypatch.setattr(api, "query_similar", lambda q, top_k: (matches, 7))

    model_answer = Answer(
        answer="High-risk systems need committee approval [ai-governance-policy].",
        confidence=0.9,
        sources_needed=True,
        citations=["ai-governance-policy#0", "invented-doc"],  # chunk id + a citation that was never retrieved
        refused=False,
    )
    monkeypatch.setattr(api, "call_model_structured", lambda messages, model: (model_answer, 100, 80, 20))

    response = client.post("/ask", json={"question": "Who approves high-risk AI?", "model": "gpt-4o-mini"})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"answer", "tokens_used", "model", "latency_ms", "cost_usd", "retrieved_chunk_ids"}
    assert body["answer"]["citations"] == ["ai-governance-policy"]  # normalised, invented one dropped
    assert body["tokens_used"] == 107  # chat + embedding tokens
    assert body["model"] == "gpt-4o-mini"
    assert body["retrieved_chunk_ids"] == ["ai-governance-policy#0", "ai-risk-assessment#2"]
    assert body["cost_usd"] > 0


def test_clean_citations_normalises_dedupes_and_drops_unknown():
    matches = [_match("ai-governance-policy", 0), _match("ai-risk-assessment", 1)]
    raw = ["[ai-governance-policy#0]", "ai-governance-policy", "ai-risk-assessment#1, made-up#3", "nist-ai-rmf"]
    assert clean_citations(raw, matches) == ["ai-governance-policy", "ai-risk-assessment"]
