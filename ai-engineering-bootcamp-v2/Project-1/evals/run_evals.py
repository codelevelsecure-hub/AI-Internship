"""Run the golden set against the real models and report pass/fail per check.

    python -m evals.run_evals                     # default model
    python -m evals.run_evals --model gpt-4o-mini
    python -m evals.run_evals --only q4-acu-measure-2-7-gap --show

Costs real OpenAI tokens (a few cents per run); prints the total.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from app import qa
from app.schemas import AskResponse

GOLDEN = Path(__file__).resolve().parent / "golden.jsonl"


def load_golden() -> list[dict[str, Any]]:
    return [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the golden set.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--only", default=None, help="Run one case by id")
    parser.add_argument("--show", action="store_true", help="Print each answer")
    args = parser.parse_args()

    cases = [c for c in load_golden() if not args.only or c["id"] == args.only]
    total_cost, passed_cases = 0.0, 0
    for case in cases:
        response = qa.answer_question(case["question"], model=args.model)
        total_cost += response.cost_usd
        checks = check_answer(case["expect"], response)
        ok = all(passed for _, passed, _ in checks)
        passed_cases += ok
        print(f"{'PASS' if ok else 'FAIL'}  {case['id']}  (${response.cost_usd:.4f}, {response.latency_ms} ms)")
        for name, passed, detail in checks:
            print(f"        {'ok ' if passed else 'XX '} {name}: {detail}")
        if args.show or not ok:
            print("        answer:", response.answer.answer.replace("\n", "\n                "))
    print(f"\n{passed_cases}/{len(cases)} cases passed, total cost ${total_cost:.4f} (model: {response.model if cases else '-'})")
    return 0 if passed_cases == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
