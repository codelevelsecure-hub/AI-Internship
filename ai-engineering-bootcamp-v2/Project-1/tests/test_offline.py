"""Offline tests for /ask: no OpenAI or Pinecone calls. Run from the Project-1 folder with `pytest`."""

import pytest
from fastapi.testclient import TestClient

import app.qa as qa
from app.api import app
from app.prompts import REFUSAL_ANSWER, clean_labels
from app.schemas import ModelAnswer
from app.search import Retrieved, named_subcategories

client = TestClient(app)


def _std(label: str, sid: str, page: int) -> dict:
    return {"id": f"nist-ai-rmf-100-1#{page}", "score": 0.8, "label": label, "corpus": "standard",
            "text": f"NIST AI 100-1, page {page}\n{sid}: Incidents and errors are communicated.",
            "metadata": {"document_id": "nist-ai-rmf-100-1", "title": "NIST AI 100-1", "subcategory_id": sid, "page": float(page)}}


def _org(label: str, doc: str, section: str) -> dict:
    return {"id": f"{doc}#0", "score": 0.7, "label": label, "corpus": "organization",
            "text": f"Acu — Policy — {section}\nHigh-Risk AI requires AI Governance Committee approval.",
            "metadata": {"document_id": doc, "title": "AI Governance Policy", "section": section}}


@pytest.fixture
def no_network(monkeypatch):
    """Fail loudly if a test reaches retrieval or the LLM without mocking them."""

    def boom(*_args, **_kwargs):
        raise AssertionError("network call attempted in an offline test")

    monkeypatch.setattr(qa, "search_both", boom)
    monkeypatch.setattr(qa, "call_model_structured", boom)


@pytest.fixture
def retrieved(monkeypatch):
    result = Retrieved([_std("S1", "MANAGE 4.3", 38)], [_org("O1", "ai-governance-policy", "6. Minimum Deployment Requirements")],
                       named_subcategories=[], embedding_tokens=7)
    monkeypatch.setattr(qa, "search_both", lambda question, top_k: result)
    return result


def _model_returns(monkeypatch, answer: ModelAnswer) -> None:
    monkeypatch.setattr(qa, "call_model_structured", lambda messages, model, schema: (answer, 100, 80, 20))


def test_empty_question_is_rejected_before_any_call(no_network):
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400
    assert "must not be empty" in response.json()["detail"]


def test_ask_splits_citations_by_corpus_and_drops_invented_labels(monkeypatch, retrieved):
    _model_returns(monkeypatch, ModelAnswer(
        answer="NIST expects incidents to be communicated [S1]. Acu requires committee approval [O1].",
        confidence=0.9, cited_labels=["S1", "S9"], gap_noted=False, refused=False,  # S9 was never shown
    ))
    response = client.post("/ask", json={"question": "Who approves high-risk AI?", "model": "gpt-4o-mini"})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"answer", "tokens_used", "model", "latency_ms", "cost_usd", "retrieved_chunk_ids"}
    std, org = body["answer"]["standard_citations"], body["answer"]["org_citations"]
    assert [c["label"] for c in std] == ["S1"] and std[0]["subcategory_id"] == "MANAGE 4.3" and std[0]["page"] == 38
    assert [c["label"] for c in org] == ["O1"]  # picked up from the answer text although not in cited_labels
    assert org[0]["section"] == "6. Minimum Deployment Requirements"
    assert body["tokens_used"] == 107 and body["cost_usd"] > 0


def test_refusal_carries_no_citations(monkeypatch, retrieved):
    _model_returns(monkeypatch, ModelAnswer(answer=REFUSAL_ANSWER, confidence=0, cited_labels=["S1"], gap_noted=True, refused=True))
    answer = client.post("/ask", json={"question": "What's the weather?"}).json()["answer"]
    assert answer["refused"] and not answer["gap_noted"]
    assert answer["standard_citations"] == [] and answer["org_citations"] == []


def test_clean_labels_normalises_dedupes_and_drops_unknown():
    assert clean_labels(["[s1]", "S1", "O2, S3", "O9; S2"], {"S1", "S2", "O2"}) == ["S1", "O2", "S2"]


@pytest.fixture
def fake_crosswalk(monkeypatch):
    import app.crosswalk as cw

    records = ({"subcategory_id": "GOVERN 1.1", "function": "Govern", "control_name": "AI Governance Policy", "source": "m.txt", "chunk_id": "crosswalk#0"},
               {"subcategory_id": "MANAGE 2.1", "function": "Manage", "control_name": "AI Incident Response", "source": "m.txt", "chunk_id": "crosswalk#1"})
    monkeypatch.setattr(cw, "load_records", lambda: records)


def test_subcategories_and_controls_endpoints(fake_crosswalk):
    subs = client.get("/frameworks/nist-ai-rmf/subcategories").json()
    assert subs["count"] == 72
    mapped = {s["id"]: s["mapped_controls"] for s in subs["subcategories"] if s["mapped_controls"]}
    assert mapped == {"GOVERN 1.1": ["AI Governance Policy"], "MANAGE 2.1": ["AI Incident Response"]}
    assert client.get("/frameworks/nist-ai-rmf/subcategories?function=govern").json()["count"] == 19
    assert client.get("/frameworks/nist-ai-rmf/subcategories?function=nope").status_code == 400
    controls = client.get("/controls").json()
    assert controls["count"] == 2


def test_evidence_validates_input_before_any_call(no_network, fake_crosswalk):
    assert client.get("/evidence").status_code == 400
    assert client.get("/evidence?subcategory=MAP%201.1&control=X").status_code == 400
    assert client.get("/evidence?subcategory=GOVERN%209.9").status_code == 404
    assert client.get("/evidence?control=Unknown").status_code == 404


def test_named_subcategories_finds_valid_ids_only():
    assert named_subcategories("Does Acu meet measure 2.7 and MANAGE-2.1? Not GOVERN 9.9.") == ["MEASURE 2.7", "MANAGE 2.1"]
