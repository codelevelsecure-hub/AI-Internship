"""Tools the agent router can call. Each tool = a Pydantic argument model (strict schema) + a handler.

Passages returned by search tools are labelled S1, S2, ... (standard) and O1, O2, ... (organization)
from one registry per question, so the final answer can cite them across several tool calls.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import openai
from pydantic import BaseModel, Field

from app import crosswalk, gap_analysis
from app.evidence import _diverse_matches, subcategory_evidence_matches
from app.frameworks import load_framework, normalize_id
from app.retrieval import embed_query, embedding_cost_usd, get_settings, query_by_vector

WEB_SEARCH_DISABLED = (
    "Live regulatory search is not enabled in v1. This version answers only from the NIST AI RMF and "
    "the organization's ingested documents; monitoring new regulations and guidance comes in v2."
)
PASSAGE_CHARS = 1500
RATIONALE_CHARS = 300


# ---------------------------------------------------------------- argument models (strict: all fields required)

class SearchStandard(BaseModel):
    """Search the NIST AI RMF 1.0 and its Playbook (what NIST expects or recommends)."""

    query: str = Field(description="What to look for, in plain words.")
    subcategory_id: str | None = Field(description="Restrict to one subcategory, e.g. 'MANAGE 4.3', or null.")


class SearchOrganization(BaseModel):
    """Search the organization's own documents: policy, procedures, minutes, risk assessments, inventory."""

    query: str = Field(description="What to look for, in plain words.")


class GetControlMapping(BaseModel):
    """The organization's control mapping: controls for a subcategory, subcategories for a control, or all (both null)."""

    subcategory_id: str | None = Field(description="e.g. 'GOVERN 1.1', or null.")
    control_name: str | None = Field(description="e.g. 'AI Incident Response', or null.")


class GetEvidence(BaseModel):
    """Organization evidence passages for ONE subcategory or ONE control (exactly one of the two)."""

    subcategory_id: str | None = Field(description="e.g. 'MEASURE 2.7', or null.")
    control_name: str | None = Field(description="e.g. 'AI Incident Response', or null.")


class RunGapAnalysis(BaseModel):
    """Assess compliance status (Covered/Partial/Missing) for many subcategories at once, with recommendations."""

    function: Literal["Govern", "Map", "Measure", "Manage"] | None = Field(description="One function, or null.")
    subcategory_ids: list[str] | None = Field(description="Specific subcategories, or null.")


class WebSearch(BaseModel):
    """Search the live web for recent or changing regulations, guidance or news beyond the ingested documents."""

    query: str = Field(description="What to look for.")


# ---------------------------------------------------------------- per-question state

@dataclass
class ToolContext:
    standards: list[dict[str, Any]] = field(default_factory=list)
    organization: list[dict[str, Any]] = field(default_factory=list)
    embedding_tokens: int = 0
    gap_cost_usd: float = 0.0
    live_search_requested: bool = False

    def label(self, matches: list[dict[str, Any]], corpus: Literal["standard", "organization"]) -> list[dict[str, Any]]:
        """Register passages (once per chunk) and return them with their labels."""

        pool = self.standards if corpus == "standard" else self.organization
        known = {m["id"]: m for m in pool}
        out = []
        for m in matches:
            if m["id"] not in known:
                m = {**m, "label": f"{'S' if corpus == 'standard' else 'O'}{len(pool) + 1}", "corpus": corpus}
                pool.append(m)
                known[m["id"]] = m
            out.append(known[m["id"]])
        return out

    def by_label(self) -> dict[str, dict[str, Any]]:
        return {m["label"]: m for m in self.standards + self.organization}

    def cost_usd(self) -> float:
        return embedding_cost_usd(self.embedding_tokens) + self.gap_cost_usd


def _passage_view(m: dict[str, Any]) -> dict[str, Any]:
    meta = m["metadata"]
    view = {"label": m["label"], "source": meta.get("title") or meta.get("source"), "text": m["text"][:PASSAGE_CHARS]}
    for key in ("subcategory_id", "page", "section", "date", "ai_system"):
        if meta.get(key) is not None:
            view[key] = meta[key]
    return view


def _known_id(raw: str | None) -> str | None:
    if not raw:
        return None
    sid = normalize_id(raw)
    load_framework().subcategory(sid)  # KeyError if unknown
    return sid


# ---------------------------------------------------------------- handlers

def search_standard(args: SearchStandard, ctx: ToolContext) -> dict[str, Any]:
    sid = _known_id(args.subcategory_id)
    text = args.query + (f"\n{sid}: {load_framework().subcategory(sid)['text']}" if sid else "")
    vector, tokens = embed_query(text)
    ctx.embedding_tokens += tokens
    flt = {"subcategory_id": {"$eq": sid}} if sid else None
    matches = query_by_vector(vector, 5, get_settings().standards_namespace, flt)
    return {"passages": [_passage_view(m) for m in ctx.label(matches, "standard")]}


def search_organization(args: SearchOrganization, ctx: ToolContext) -> dict[str, Any]:
    vector, tokens = embed_query(args.query)
    ctx.embedding_tokens += tokens
    return {"passages": [_passage_view(m) for m in ctx.label(_diverse_matches(vector, 5), "organization")]}


def get_control_mapping(args: GetControlMapping, ctx: ToolContext) -> dict[str, Any]:
    if args.subcategory_id:
        sid = _known_id(args.subcategory_id)
        records = crosswalk.controls_for(sid)
        return {"subcategory_id": sid, "mapped_controls": [r["control_name"] for r in records],
                "note": None if records else "No organization control is mapped to this subcategory."}
    if args.control_name:
        records = crosswalk.subcategories_for(args.control_name)
        return {"control_name": args.control_name, "subcategory_ids": [r["subcategory_id"] for r in records],
                "note": None if records else "This control is not in the organization's control mapping."}
    return {"controls": crosswalk.controls(), "subcategories_without_mapping": 72 - len({r["subcategory_id"] for r in crosswalk.load_records()})}


def get_evidence(args: GetEvidence, ctx: ToolContext) -> dict[str, Any]:
    framework = load_framework()
    if args.subcategory_id:
        sub = framework.subcategory(_known_id(args.subcategory_id))
        controls = [r["control_name"] for r in crosswalk.controls_for(sub["id"])]
        matches, _vector, tokens = subcategory_evidence_matches(sub, controls, 5)
        ctx.embedding_tokens += tokens
        return {"subcategory": {"id": sub["id"], "text": sub["text"]}, "mapped_controls": controls,
                "evidence_basis": "mapped controls + topic similarity" if controls else "topic similarity only (no mapped control)",
                "passages": [_passage_view(m) for m in ctx.label(matches, "organization")]}
    if args.control_name:
        records = crosswalk.subcategories_for(args.control_name)
        if not records:
            return {"error": f"'{args.control_name}' is not in the organization's control mapping."}
        subs = [framework.subcategory(r["subcategory_id"]) for r in records]
        vector, tokens = embed_query("\n".join([f"Control: {records[0]['control_name']}"] + [f"{s['id']}: {s['text']}" for s in subs]))
        ctx.embedding_tokens += tokens
        return {"control_name": records[0]["control_name"], "subcategory_ids": [s["id"] for s in subs],
                "passages": [_passage_view(m) for m in ctx.label(_diverse_matches(vector, 5), "organization")]}
    return {"error": "Pass subcategory_id or control_name."}


def run_gap_analysis(args: RunGapAnalysis, ctx: ToolContext) -> dict[str, Any]:
    report = gap_analysis.run_gap_analysis(function=args.function, subcategory_ids=args.subcategory_ids or None)
    ctx.gap_cost_usd += report.cost_usd
    return {
        "scope": report.scope,
        "counts": report.counts,
        "findings": [
            {"subcategory_id": f.subcategory_id, "status": f.status, "mapped_controls": f.mapped_controls,
             "rationale": f.rationale[:RATIONALE_CHARS],
             "top_recommendations": [f"{r.priority}: {r.action}" for r in f.recommendations[:2]]}
            for f in report.findings
        ],
        "note": "Statuses come from the gap analysis; cite them as 'per the gap analysis'.",
    }


def web_search(args: WebSearch, ctx: ToolContext) -> dict[str, Any]:
    ctx.live_search_requested = True
    return {"enabled": False, "message": WEB_SEARCH_DISABLED}


# ---------------------------------------------------------------- registry

@dataclass(frozen=True)
class Tool:
    name: str
    args_model: type[BaseModel]
    handler: Callable[[Any, ToolContext], dict[str, Any]]


TOOLS: dict[str, Tool] = {
    t.name: t for t in (
        Tool("search_standard", SearchStandard, search_standard),
        Tool("search_organization", SearchOrganization, search_organization),
        Tool("get_control_mapping", GetControlMapping, get_control_mapping),
        Tool("get_evidence", GetEvidence, get_evidence),
        Tool("run_gap_analysis", RunGapAnalysis, run_gap_analysis),
        Tool("web_search", WebSearch, web_search),
    )
}


def openai_tool_definitions() -> list[dict[str, Any]]:
    return [openai.pydantic_function_tool(t.args_model, name=t.name, description=t.args_model.__doc__) for t in TOOLS.values()]


def run_tool(name: str, arguments_json: str, ctx: ToolContext) -> dict[str, Any]:
    """Validate arguments and run a tool. Errors are returned to the model as data, never raised."""

    tool = TOOLS.get(name)
    if tool is None:
        return {"error": f"Unknown tool '{name}'."}
    try:
        args = tool.args_model.model_validate_json(arguments_json)
        return tool.handler(args, ctx)
    except KeyError as exc:
        return {"error": f"Unknown identifier: {exc.args[0]}"}
    except Exception as exc:  # validation or service failure — let the model adapt or explain
        return {"error": f"{type(exc).__name__}: {exc}"}
