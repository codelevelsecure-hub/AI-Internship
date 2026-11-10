"""Offline test of the agent loop: a scripted fake OpenAI client and mocked retrieval."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.agents.router as router
import app.agents.tools as tools
from app.api import app
from app.schemas import AgentAnswer

client = TestClient(app)


def _completion(message):
    usage = SimpleNamespace(total_tokens=100, prompt_tokens=80, completion_tokens=20)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def _tool_call(call_id: str, name: str, args: dict):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


class FakeClient:
    """Returns the scripted messages in order and records what the agent sent."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=self.parse))

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return _completion(self.script.pop(0))


@pytest.fixture
def org_passages(monkeypatch):
    match = {"id": "ai-governance-policy#4", "score": 0.7,
             "text": "Acu — AI Governance Policy — 6. Minimum Deployment Requirements\nHigh-Risk AI requires committee approval.",
             "metadata": {"document_id": "ai-governance-policy", "title": "AI Governance Policy", "section": "6. Minimum Deployment Requirements"}}
    monkeypatch.setattr(tools, "embed_query", lambda text: ([0.1], 5))
    monkeypatch.setattr(tools, "_diverse_matches", lambda vector, k: [dict(match)])


def test_agent_runs_tool_then_cites_its_passages(monkeypatch, org_passages):
    fake = FakeClient([
        SimpleNamespace(content=None, parsed=None, tool_calls=[
            _tool_call("c1", "search_organization", {"query": "high-risk approval"}),
            _tool_call("c2", "web_search", {"query": "EU AI Act"}),
        ]),
        SimpleNamespace(content="{}", tool_calls=None, parsed=AgentAnswer(
            answer="The AI Governance Committee approves high-risk AI [O1]. Live search is not enabled.",
            confidence=0.9, cited_labels=["O1", "S4"], gap_noted=False, refused=False, needs_live_search=False)),
    ])
    monkeypatch.setattr(router, "_client", lambda: fake)

    body = client.post("/ask", json={"question": "Who approves high-risk AI, and what's new in the EU AI Act?"}).json()
    assert body["mode"] == "agent"
    assert [t["tool"] for t in body["tool_trace"]] == ["search_organization", "web_search"]
    assert body["tool_trace"][0]["summary"] == "1 passages: O1"
    answer = body["answer"]
    assert [c["label"] for c in answer["org_citations"]] == ["O1"] and answer["standard_citations"] == []  # S4 never shown
    assert answer["needs_live_search"] is True  # set because web_search was called, even though the model said False
    assert body["retrieved_chunk_ids"] == ["ai-governance-policy#4"]
    # The second call carried the tool results back to the model.
    sent = fake.calls[1]["messages"]
    assert [m["role"] for m in sent[-3:]] == ["assistant", "tool", "tool"]
    assert "Live regulatory search is not enabled" in sent[-1]["content"]


def test_agent_stops_after_max_steps_and_forbids_tools(monkeypatch, org_passages):
    looping = [SimpleNamespace(content=None, parsed=None, tool_calls=[_tool_call(f"c{i}", "search_organization", {"query": "x"})])
               for i in range(router.MAX_STEPS)]
    final = SimpleNamespace(content="{}", tool_calls=None, parsed=AgentAnswer(
        answer="Done [O1].", confidence=0.5, cited_labels=["O1"], gap_noted=False, refused=False, needs_live_search=False))
    fake = FakeClient(looping + [final])
    monkeypatch.setattr(router, "_client", lambda: fake)

    response = router.run_agent("Loop forever?")
    assert len(response.tool_trace) == router.MAX_STEPS
    assert fake.calls[-1]["tool_choice"] == "none"
    assert response.answer.answer == "Done [O1]."


def test_unknown_tool_and_bad_arguments_return_errors_not_exceptions():
    ctx = tools.ToolContext()
    assert "Unknown tool" in tools.run_tool("delete_everything", "{}", ctx)["error"]
    assert "ValidationError" in tools.run_tool("search_organization", '{"wrong": 1}', ctx)["error"]
