"""Offline tests for gap analysis: OpenAI, Pinecone and the crosswalk are mocked; the cache is a temp file."""

import pytest

import app.gap_analysis as gap
from app.schemas import GapAssessment, ModelRecommendation


def _ev(i: int) -> dict:
    return {"id": f"ai-governance-policy#{i}", "score": 0.6, "text": f"Acu — AI Governance Policy — Section {i}\nEvidence text {i}.",
            "metadata": {"document_id": "ai-governance-policy", "title": "AI Governance Policy", "section": f"Section {i}", "doc_type": "policy"}}


def _pb(i: int) -> dict:
    return {"id": f"nist-ai-rmf-playbook#{i}", "score": 0.7, "text": f"Playbook — Suggested Actions\nAction {i}.",
            "metadata": {"page": float(40 + i), "subcategory_id": "GOVERN 1.1"}}


@pytest.fixture
def mocked(monkeypatch, tmp_path):
    calls = {"model": 0}
    monkeypatch.setattr(gap, "DATA_DIR", tmp_path)
    monkeypatch.setattr(gap, "CACHE_FILE", tmp_path / "gap_cache.json")
    monkeypatch.setattr(gap.crosswalk, "controls_for", lambda sid: [{"control_name": "AI Governance Policy"}] if sid == "GOVERN 1.1" else [])
    monkeypatch.setattr(gap, "subcategory_evidence_matches", lambda sub, controls, k: ([_ev(1), _ev(2)], [0.1] * 3, 12))
    monkeypatch.setattr(gap, "query_by_vector", lambda vector, k, ns, flt: [_pb(1), _pb(2)])
    monkeypatch.setattr(gap, "get_settings", lambda: type("S", (), {"standards_namespace": "std", "org_namespace": "org"})())
    monkeypatch.setattr(gap, "organization_profile", lambda: "Acu profile")

    def set_answer(answer: GapAssessment):
        def fake_call(messages, model, schema):
            calls["model"] += 1
            return answer, 500, 400, 100
        monkeypatch.setattr(gap, "call_model_structured", fake_call)

    return calls, set_answer


def test_finding_keeps_only_shown_labels_and_is_cached(mocked):
    calls, set_answer = mocked
    set_answer(GapAssessment(
        status="Partial", rationale="Policy states intent [E1].", evidence_labels=["E1", "E7"],  # E7 never shown
        mapping_assessment="Partly fits.", confidence=0.8,
        recommendations=[ModelRecommendation(action="Document legal requirements.", priority="High",
                                             owner_hint="AI Governance Committee", playbook_labels=["P2", "P9"])],
    ))
    first = gap.assess_subcategory("govern 1.1")
    assert first.subcategory_id == "GOVERN 1.1" and first.mapped_controls == ["AI Governance Policy"]
    assert [e.label for e in first.evidence] == ["E1"] and first.evidence[0].section == "Section 1"
    assert [(p.label, p.page) for p in first.recommendations[0].playbook_refs] == [("P2", 42)]
    assert not first.cached and first.cost_usd > 0

    second = gap.assess_subcategory("GOVERN 1.1")
    assert second.cached and calls["model"] == 1  # no second model call
    assert second.status == "Partial"

    gap.assess_subcategory("GOVERN 1.1", refresh=True)
    assert calls["model"] == 2  # refresh bypasses the cache


def test_covered_drops_recommendations_and_latest_report(mocked):
    _calls, set_answer = mocked
    set_answer(GapAssessment(
        status="Covered", rationale="Done [E2].", evidence_labels=["E2"], mapping_assessment="Fits.", confidence=0.9,
        recommendations=[ModelRecommendation(action="x", priority="Low", owner_hint="y", playbook_labels=[])],
    ))
    report = gap.run_gap_analysis(subcategory_ids=["GOVERN 1.1", "MAP 1.1"])
    assert report.counts["Covered"] == 2 and all(f.recommendations == [] for f in report.findings)
    assert report.counts_by_function == {"Govern": {"Covered": 1, "Partial": 0, "Missing": 0, "Not applicable": 0},
                                         "Map": {"Covered": 1, "Partial": 0, "Missing": 0, "Not applicable": 0}}
    latest = gap.latest_report()
    assert [f.subcategory_id for f in latest.findings] == ["GOVERN 1.1", "MAP 1.1"] and latest.cost_usd == 0


def test_unknown_subcategory_raises_before_any_call(mocked):
    with pytest.raises(KeyError):
        gap.run_gap_analysis(subcategory_ids=["GOVERN 9.9"])
