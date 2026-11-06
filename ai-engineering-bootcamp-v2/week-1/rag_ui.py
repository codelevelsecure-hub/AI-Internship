"""Session 2 — minimal Streamlit UI for the RAG API.

A thin client: every action is an HTTP call to the FastAPI service (/ingest, /ask,
/health/pinecone). No retrieval, prompting, or OpenAI/Pinecone calls happen here,
so the API stays the single source of truth. The UI needs no API keys.

Run (from this folder):
    streamlit run rag_ui.py
Point it at Render instead of localhost (PowerShell):
    $env:RAG_API_URL = "https://ai-internship-76vl.onrender.com"; streamlit run rag_ui.py
or just change the URL in the sidebar.
"""

import json
import os

import httpx
import streamlit as st

DEFAULT_API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")
TIMEOUT_S = 120.0  # Render's free plan can take ~60s to wake from sleep.

st.set_page_config(page_title="Northwind RAG", layout="wide")


def call_api(method: str, base_url: str, path: str, payload: dict | None = None) -> tuple[int | None, dict | str]:
    """Return (status_code, json-or-text). status_code is None if the API couldn't be reached."""

    try:
        r = httpx.request(method, f"{base_url.rstrip('/')}{path}", json=payload, timeout=TIMEOUT_S)
    except httpx.HTTPError as exc:
        return None, f"Could not reach {base_url}: {exc}"
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, r.text


def show_error(status: int | None, body: dict | str) -> None:
    """Show the API's own error message (FastAPI puts it in 'detail')."""

    detail = body.get("detail", body) if isinstance(body, dict) else body
    st.error(f"{'Connection error' if status is None else f'HTTP {status}'}: {detail}")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("API")
    api_url = st.text_input("API base URL", DEFAULT_API_URL, help="Set RAG_API_URL to change the default.")
    if st.button("Check connection"):
        with st.spinner("Calling /health/pinecone (first call may wake the server)..."):
            status, body = call_api("GET", api_url, "/health/pinecone")
        if status == 200 and isinstance(body, dict) and body.get("ok"):
            st.success(f"Connected - index `{body['index']}`, {body['total_vector_count']} chunks")
        else:
            show_error(status, body)

    st.header("Ask settings")
    model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini"])
    top_k = st.slider("Chunks to retrieve (top_k)", 1, 10, 5)

st.title("Northwind Policy Assistant")
st.caption(f"RAG demo - answers come only from documents ingested into the API at `{api_url}`.")

ask_tab, ingest_tab = st.tabs(["Ask a question", "Ingest a document"])

# ---------------------------------------------------------------- ask
with ask_tab:
    with st.form("ask_form"):
        question = st.text_input("Question", placeholder="e.g. What is the mileage rate for business travel?")
        asked = st.form_submit_button("Ask", type="primary")

    if asked:
        with st.spinner("Retrieving chunks and generating an answer..."):
            status, body = call_api("POST", api_url, "/ask", {"question": question, "model": model, "top_k": top_k})

        if status != 200 or not isinstance(body, dict):
            show_error(status, body)
        else:
            answer = body["answer"]
            if answer.get("refused"):
                st.warning(f"**Refused - not in the documents**\n\n{answer['answer']}", icon="🚫")
            else:
                st.success(f"**Answer**\n\n{answer['answer']}", icon="✅")

            st.markdown("**Cited documents:** " + (
                " ".join(f"`{doc_id}`" for doc_id in answer.get("citations", [])) or "_none_"
            ))

            cols = st.columns(5)
            cols[0].metric("Refused", "Yes" if answer.get("refused") else "No")
            cols[1].metric("Confidence", f"{answer.get('confidence', 0):.2f}")
            cols[2].metric("Tokens", body["tokens_used"])
            cols[3].metric("Cost (USD)", f"${body['cost_usd']:.4f}")
            cols[4].metric("Latency", f"{body['latency_ms']} ms")

            st.markdown("**Retrieved chunks** (best match first)")
            st.dataframe(
                [{"rank": i, "chunk_id": cid} for i, cid in enumerate(body.get("retrieved_chunk_ids", []), start=1)],
                hide_index=True,
            )

            with st.expander("Raw JSON response"):
                st.code(json.dumps(body, indent=2), language="json")

# ---------------------------------------------------------------- ingest
with ingest_tab:
    st.caption("Re-ingesting an existing document_id replaces that document's chunks.")
    with st.form("ingest_form"):
        document_id = st.text_input("document_id", placeholder="e.g. doc7_travel_faq")
        source = st.text_input("source (optional)", placeholder="e.g. travel_faq.txt")
        text = st.text_area("Text", height=250, placeholder="Paste the document text here...")
        ingested = st.form_submit_button("Ingest", type="primary")

    if ingested:
        # No client-side validation: the API decides what's valid and returns a clear 400.
        payload = {"document_id": document_id, "text": text}
        if source.strip():
            payload["source"] = source
        with st.spinner("Chunking, embedding, and upserting..."):
            status, body = call_api("POST", api_url, "/ingest", payload)

        if status == 200 and isinstance(body, dict):
            st.success(f"Indexed **{body['chunks_indexed']}** chunk(s) for `{body['document_id']}` - status: {body['status']}")
            st.code(json.dumps(body, indent=2), language="json")
        else:
            show_error(status, body)
