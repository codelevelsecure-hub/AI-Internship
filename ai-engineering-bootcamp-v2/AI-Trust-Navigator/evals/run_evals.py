"""Evaluate AI Trust Navigator against the golden sets and report four metrics.

    python -m evals.run_evals                         # agent mode (POST /ask), default model
    python -m evals.run_evals --mode simple --model gpt-4o-mini
    python -m evals.run_evals --only q4-acu-measure-2-7-gap --show
    python -m evals.run_evals --skip-gaps             # question cases only

Metrics
  retrieval hit rate   — search finds the expected subcategory / document (cases that name one)
  citation accuracy    — the answer passes every expectation of its case (citations, gap flag, text)
  gap agreement        — gap status is one of the accepted statuses in evals/gap_golden.jsonl
  refusal correctness  — off-topic and injection questions are refused; on-topic ones are not
The run fails (exit 1) when a metric is below its threshold. Costs real tokens (cents); prints the total.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from app import qa
from app.agents import router
from app.gap_analysis import run_gap_analysis
from app.schemas import AskResponse
from app.search import search_both

HERE = Path(__file__).resolve().parent
GOLDEN, GAP_GOLDEN = HERE / "golden.jsonl", HERE / "gap_golden.jsonl"
THRESHOLDS = {"retrieval hit rate": 0.90, "citation accuracy": 0.80, "gap agreement": 0.75, "refusal correctness": 1.00}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_answer(expect: dict[str, Any], response: AskResponse) -> list[tuple[str, bool, str]]:
    """Return (check name, passed, detail) for every expectation present in the case."""

    a = response.answer
    std_ids = {c.subcategory_id for c in a.standard_citations if c.subcategory_id}
    org_docs = {c.document_id for c in a.org_citations}
    checks: list[tuple[str, bool, str]] = []
    if "refused" in expect:
        checks.append(("refused", a.refused == expect["refused"], f"refused={a.refused}"))
    if "gap_noted" in expect:
        checks.append(("gap_noted", a.gap_noted == expect["gap_noted"], f"gap_noted={a.gap_noted}"))
    if "needs_live_search" in expect:
        checks.append(("needs_live_search", a.needs_live_search == expect["needs_live_search"], f"needs_live_search={a.needs_live_search}"))
    if "standard_subcategories_any" in expect:
        hit = std_ids & set(expect["standard_subcategories_any"])
        checks.append(("standard citation", bool(hit), f"cited {sorted(std_ids) or 'none'}"))
    if "min_standard_citations" in expect:
        n = len(a.standard_citations)
        checks.append(("standard count", n >= expect["min_standard_citations"], f"{n} standard citations"))
    if "org_documents_any" in expect:
        hit = org_docs & set(expect["org_documents_any"])
        checks.append(("organization citation", bool(hit), f"cited {sorted(org_docs) or 'none'}"))
    if "answer_contains_any" in expect:
        found = [p for p in expect["answer_contains_any"] if p.lower() in a.answer.lower()]
        checks.append(("answer text", bool(found), f"found {found or 'none of ' + str(expect['answer_contains_any'])}"))
    return checks


def retrieval_hit(case: dict[str, Any]) -> tuple[bool, str] | None:
    """Does one retrieval pass surface what the case expects? None when the case expects nothing to retrieve."""

    expect = case["expect"]
    want_subs, want_docs = set(expect.get("standard_subcategories_any", [])), set(expect.get("org_documents_any", []))
    if not want_subs and not want_docs:
        return None
    retrieved = search_both(case["question"])
    subs = {m["metadata"].get("subcategory_id") for m in retrieved.standards}
    docs = {m["metadata"].get("document_id") for m in retrieved.organization}
    ok = (not want_subs or bool(subs & want_subs)) and (not want_docs or bool(docs & want_docs))
    return ok, f"retrieved subcategories {sorted(s for s in subs if s)[:6]}, documents {sorted(d for d in docs if d)}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the golden sets and report metrics.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--mode", choices=["agent", "simple"], default="agent", help="agent = /ask, simple = /ask/simple")
    parser.add_argument("--only", default=None, help="Run one question case by id")
    parser.add_argument("--show", action="store_true", help="Print each answer")
    parser.add_argument("--skip-gaps", action="store_true", help="Skip the gap-agreement set")
    args = parser.parse_args()

    cases = [c for c in load_jsonl(GOLDEN) if not args.only or c["id"] == args.only]
    if args.mode == "simple":  # simple mode has no web-search tool, so it can't flag live search
        cases = [c for c in cases if c["kind"] != "live_search"]

    total_cost = 0.0
    retrieval, citation, refusal = [], [], []
    print(f"== Question cases ({args.mode} mode)")
    for case in cases:
        hit = retrieval_hit(case)
        if hit is not None:
            retrieval.append(hit[0])
        if args.mode == "agent":
            response = router.run_agent(case["question"], model=args.model)
        else:
            response = qa.answer_question(case["question"], model=args.model)
        total_cost += response.cost_usd
        checks = check_answer(case["expect"], response)
        ok = all(passed for _, passed, _ in checks)
        citation.append(ok)
        if "refused" in case["expect"]:
            refusal.append(response.answer.refused == case["expect"]["refused"])
        tools = f", tools: {[t.tool for t in response.tool_trace]}" if response.tool_trace else ""
        print(f"{'PASS' if ok else 'FAIL'}  {case['id']}  (${response.cost_usd:.4f}, {response.latency_ms} ms{tools})")
        if hit is not None:
            print(f"        {'ok ' if hit[0] else 'XX '} retrieval: {hit[1]}")
        for name, passed, detail in checks:
            print(f"        {'ok ' if passed else 'XX '} {name}: {detail}")
        if args.show or not ok:
            print("        answer:", response.answer.answer.replace("\n", "\n                "))

    gap_agreement: list[bool] = []
    if not args.skip_gaps and not args.only:
        print("\n== Gap agreement (evals/gap_golden.jsonl; cached findings cost nothing)")
        golden_gaps = load_jsonl(GAP_GOLDEN)
        report = run_gap_analysis(subcategory_ids=[g["subcategory_id"] for g in golden_gaps])
        total_cost += report.cost_usd
        status = {f.subcategory_id: f.status for f in report.findings}
        for g in golden_gaps:
            ok = status[g["subcategory_id"]] in g["accept"]
            gap_agreement.append(ok)
            note = "" if g.get("reviewed") else "  (proposed, not yet reviewed)"
            print(f"{'PASS' if ok else 'FAIL'}  {g['subcategory_id']:<12} got {status[g['subcategory_id']]:<8} accept {g['accept']}{note}")

    metrics = {
        "retrieval hit rate": retrieval,
        "citation accuracy": citation,
        "gap agreement": gap_agreement,
        "refusal correctness": refusal,
    }
    print("\n== Metrics")
    failed = False
    for name, results in metrics.items():
        if not results:
            print(f"  {name:<20} n/a")
            continue
        rate = sum(results) / len(results)
        below = rate < THRESHOLDS[name]
        failed |= below
        print(f"  {name:<20} {rate:6.1%}  ({sum(results)}/{len(results)})  threshold {THRESHOLDS[name]:.0%}  {'BELOW' if below else 'ok'}")
    print(f"\nTotal cost ${total_cost:.4f} (mode: {args.mode}, model: {args.model or 'default'})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
