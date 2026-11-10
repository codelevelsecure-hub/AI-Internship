"""AI Trust Navigator — Streamlit UI.

A thin client: every action is an HTTP call to the FastAPI service. No retrieval, prompting, or
OpenAI/Pinecone calls happen here, so the API stays the single source of truth.

Run (from this folder, with the API running):
    streamlit run ui.py --server.port 8502
Point it at another server (PowerShell):
    $env:ATN_API_URL = "https://<your-service>.onrender.com"; streamlit run ui.py
"""

import json
import os

import httpx
import streamlit as st

DEFAULT_API_URL = os.getenv("ATN_API_URL", "http://127.0.0.1:8001")
TIMEOUT_S = 180.0  # Render's free plan can take ~60s to wake; gap runs take longer.

st.set_page_config(page_title="AI Trust Navigator", layout="wide")


def call_api(method: str, path: str, payload: dict | None = None, params: dict | None = None) -> tuple[int | None, dict | list | str]:
    """Return (status_code, json-or-text). status_code is None if the API couldn't be reached."""

    try:
        r = httpx.request(method, f"{st.session_state.api_url.rstrip('/')}{path}", json=payload, params=params, timeout=TIMEOUT_S)
    except httpx.HTTPError as exc:
        return None, f"Could not reach {st.session_state.api_url}: {exc}"
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, r.text


def show_error(status: int | None, body: dict | list | str) -> None:
    detail = body.get("detail", body) if isinstance(body, dict) else body
    st.error(f"{'Connection error' if status is None else f'HTTP {status}'}: {detail}")


def show_citations(title: str, citations: list[dict]) -> None:
    st.markdown(f"**{title}**")
    if not citations:
        st.caption("None cited.")
        return
    for c in citations:
        where = " · ".join(str(x) for x in (c.get("subcategory_id"), f"p. {c['page']}" if c.get("page") else None, c.get("section")) if x)
        with st.expander(f"[{c['label']}] {c['title']}" + (f" — {where}" if where else "")):
            st.write(c["quote"])
            st.caption(f"chunk: {c['chunk_id']}")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("API")
    st.session_state.api_url = st.text_input("API base URL", DEFAULT_API_URL, help="Set ATN_API_URL to change the default.")
    if st.button("Check connection"):
        with st.spinner("Calling /health/pinecone (first call may wake the server)..."):
            status, body = call_api("GET", "/health/pinecone")
        if status == 200 and isinstance(body, dict) and body.get("ok"):
            ns = body["namespaces"]
            st.success(f"Connected — standards: {ns['standards']['vector_count']} chunks, "
                       f"organization: {ns['organization']['vector_count']} chunks")
        else:
            show_error(status, body)

    st.header("Settings")
    model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini"])
    top_k = st.slider("Passages per corpus (top_k)", 1, 10, 5)

st.title("AI Trust Navigator")
st.caption("Compares the organization's AI governance documents with the NIST AI RMF 1.0. "
           "Answers come only from the ingested documents.")

ask_tab, evidence_tab, ingest_tab = st.tabs(["Ask", "Evidence explorer", "Ingest a document"])


@st.cache_data(ttl=300, show_spinner=False)
def load_lists(api_url: str) -> tuple[list[dict], list[dict]] | None:
    """Subcategories and controls for the pickers (cached per API URL for 5 minutes)."""

    s_status, subs = call_api("GET", "/frameworks/nist-ai-rmf/subcategories")
    c_status, ctrls = call_api("GET", "/controls")
    if s_status != 200 or c_status != 200:
        return None
    return subs["subcategories"], ctrls["controls"]


def show_evidence(items: list[dict]) -> None:
    if not items:
        st.info("No evidence passages found.")
    for e in items:
        label = f"{e['title']} — {e['section']}" + (f" ({e['date']})" if e.get("date") else "")
        with st.expander(f"{label}  ·  score {e['score']:.2f}"):
            st.write(e["quote"])
            st.caption(f"chunk: {e['chunk_id']}" + (f" · system: {e['ai_system']}" if e.get("ai_system") else ""))

# ---------------------------------------------------------------- ask
with ask_tab:
    with st.form("ask_form"):
        question = st.text_input("Question", placeholder="e.g. Does Acu address MEASURE 2.7?")
        asked = st.form_submit_button("Ask", type="primary")

    if asked:
        with st.spinner("Retrieving from both corpora and generating an answer..."):
            status, body = call_api("POST", "/ask", {"question": question, "model": model, "top_k": top_k})
        if status != 200 or not isinstance(body, dict):
            show_error(status, body)
        else:
            answer = body["answer"]
            if answer["refused"]:
                st.warning(f"**Not covered by the documents**\n\n{answer['answer']}")
            elif answer["gap_noted"]:
                st.warning(f"**Gap noted**\n\n{answer['answer']}")
            else:
                st.success(answer["answer"])

            left, right = st.columns(2)
            with left:
                show_citations("NIST AI RMF (standard)", answer["standard_citations"])
            with right:
                show_citations("Organization documents", answer["org_citations"])

            cols = st.columns(4)
            cols[0].metric("Confidence", f"{answer['confidence']:.2f}")
            cols[1].metric("Tokens", body["tokens_used"])
            cols[2].metric("Cost (USD)", f"${body['cost_usd']:.4f}")
            cols[3].metric("Latency", f"{body['latency_ms']} ms")
            with st.expander("Raw JSON response"):
                st.code(json.dumps(body, indent=2), language="json")

# ---------------------------------------------------------------- evidence
with evidence_tab:
    st.caption("Finds the organization's documents that bear on a NIST subcategory or on one of its controls. "
               "No LLM call: these are retrieved passages, not a judgement.")
    lists = load_lists(st.session_state.api_url)
    if lists is None:
        st.error("Could not load subcategories and controls. Check the API connection in the sidebar.")
    else:
        subcategories, controls = lists
        mode = st.radio("Look up by", ["NIST subcategory", "Organization control"], horizontal=True)
        if mode == "NIST subcategory":
            options = {f"{s['id']} — {s['text'][:90]}" + (" ✓ mapped" if s["mapped_controls"] else ""): s["id"] for s in subcategories}
            choice = st.selectbox("Subcategory", list(options))
            params = {"subcategory": options[choice]}
        else:
            choice = st.selectbox("Control", [c["control_name"] for c in controls])
            params = {"control": choice}
        if st.button("Find evidence", type="primary"):
            status, body = call_api("GET", "/evidence", params=params)
            if status != 200 or not isinstance(body, dict):
                show_error(status, body)
            elif "subcategory" in body:
                sub = body["subcategory"]
                st.markdown(f"**{sub['id']}** ({sub['function']}, AI RMF 1.0 p. {sub['page']}): {sub['text']}")
                if body["mapping_status"] == "mapped":
                    st.success("Mapped to: " + ", ".join(body["mapped_controls"]))
                else:
                    st.warning("No organization control is mapped to this subcategory. "
                               "Passages below are related by topic only.")
                show_evidence(body["evidence"])
            else:
                st.markdown("**Mapped to:** " + ", ".join(f"{s['id']} — {s['text']}" for s in body["subcategories"]))
                show_evidence(body["evidence"])

# ---------------------------------------------------------------- ingest
with ingest_tab:
    st.caption("Adds an ad-hoc document to the organization corpus. For the full corpus, run "
               "`python ingest_docs.py` instead. Re-ingesting an existing document_id replaces it.")
    with st.form("ingest_form"):
        document_id = st.text_input("document_id", placeholder="e.g. ai-vendor-policy")
        source = st.text_input("source (optional)", placeholder="e.g. AI Vendor Policy.txt")
        text = st.text_area("Text", height=250, placeholder="Paste the document text here...")
        ingested = st.form_submit_button("Ingest", type="primary")
    if ingested:
        payload = {"document_id": document_id, "text": text}
        if source.strip():
            payload["source"] = source
        with st.spinner("Chunking, embedding, and upserting..."):
            status, body = call_api("POST", "/ingest", payload)
        if status == 200 and isinstance(body, dict):
            st.success(f"Indexed **{body['chunks_indexed']}** chunk(s) for `{body['document_id']}`")
        else:
            show_error(status, body)
