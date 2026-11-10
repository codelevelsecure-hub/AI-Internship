# AI Trust Navigator — v1

An AI governance assistant that compares an organization's own AI policies, procedures and evidence
with the **NIST AI Risk Management Framework (AI RMF 1.0)**. It answers compliance questions with
citations to both sides, finds evidence, runs a gap analysis across all 72 NIST subcategories, and
recommends next steps grounded in the NIST Playbook.

v1 covers one organization and one framework. Live monitoring of new regulations (web search) is
built in as a disabled tool and switches on in v2.

## How it works

```
corpus/external/  NIST AI RMF 100-1 + Playbook (PDF) ──► standards namespace      ┐
corpus/internal/  organization documents (text)      ──► organization namespace   ├─ one Pinecone index
                  Control Mapping rows               ──► crosswalk records        ┘
frameworks/nist_ai_rmf_1.0.json   the 72 subcategories (source of truth for IDs)

POST /ask        agent: picks tools (search NIST, search org docs, control mapping, evidence,
                 gap analysis, web search*) and answers with citations to both corpora
POST /ask/simple one retrieval pass over both corpora, no tools (cheaper, faster)
POST /gaps       Covered / Partial / Missing per subcategory, with recommendations (cached)
GET  /evidence   organization evidence for one subcategory or one control (no LLM call)
                 * web_search returns "not enabled in v1"
```

Answers cite passages by label (`S1` = NIST, `O1` = organization); the server turns labels into full
citations (document, subcategory, page, section), so the model cannot invent a source. A control
mapping is treated as a claim to check, not as evidence.

## Setup (Windows PowerShell, from this folder)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # then fill in OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME
```

One-time Pinecone index (1536 dimensions, cosine; skipped if it already exists):

```powershell
python create_pinecone_index.py
```

## Add the documents and ingest

1. Put the NIST PDFs in `corpus/external/` and the organization's `.txt`/`.md` files in
   `corpus/internal/`.
2. List every file in that folder's `manifest.json` (`document_id`, `doc_type`; optional `date`,
   `ai_system`, `stop_at`). Unlisted files are skipped with a warning.
3. Ingest (writes straight to Pinecone; the laptop and a deployed server share the same data):

```powershell
python ingest_docs.py --dry-run     # files, namespaces, chunk counts — sends nothing
python ingest_docs.py               # both corpora (about $0.002 of embeddings)
python ingest_docs.py --corpus internal   # after editing an organization document
```

Re-running replaces each document's chunks; nothing is duplicated. `corpus/internal/` is
git-ignored: organization documents never go into the repository.

## Run

Two terminals, both with the venv active:

```powershell
uvicorn main:app --host 127.0.0.1 --port 8001 --reload
```
```powershell
$env:ATN_API_URL = "http://127.0.0.1:8001"; streamlit run ui.py --server.port 8502
```

Open http://localhost:8502 → **Check connection**. Tabs: **Ask**, **Gap analysis**,
**Evidence explorer**, **Ingest a document**. API docs: http://127.0.0.1:8001/docs.

If `API_KEY` is set on the server, every route except `/health/pinecone` and the docs needs the
header `X-API-Key: <key>`. In the UI, enter it in the sidebar or set `$env:ATN_API_KEY` first.

## Example requests (PowerShell)

```powershell
$h = @{ "X-API-Key" = $env:API_KEY }   # omit -Headers when API_KEY is not set
Invoke-RestMethod -Method Post http://127.0.0.1:8001/ask -Headers $h -ContentType "application/json" -Body '{"question": "Does Acu address MEASURE 2.7?"}'
Invoke-RestMethod -Method Post http://127.0.0.1:8001/gaps -Headers $h -ContentType "application/json" -Body '{"function": "Govern"}'
Invoke-RestMethod "http://127.0.0.1:8001/evidence?subcategory=MANAGE%204.3" -Headers $h
Invoke-RestMethod http://127.0.0.1:8001/controls -Headers $h
```

## Test and evaluate

```powershell
python -m pytest -q                 # offline: no OpenAI or Pinecone calls, ~2 s
python -m evals.run_evals           # live: golden questions + gap agreement, ~$0.08 (agent mode)
python -m evals.run_evals --mode simple --model gpt-4o-mini   # ~$0.004
```

The eval run reports retrieval hit rate, citation accuracy, gap agreement and refusal correctness
against thresholds (90 / 80 / 75 / 100 %) and exits non-zero if any metric is below. Golden sets:
`evals/golden.jsonl` (questions) and `evals/gap_golden.jsonl` (accepted gap statuses; set
`"reviewed": true` once a person has confirmed an entry).

## Costs (OpenAI list prices, measured)

| Action | Model | Cost |
|---|---|---|
| Ingest both corpora | text-embedding-3-small | ~$0.002 |
| `/ask` (agent) | gpt-4o | ~$0.01 per question |
| `/ask/simple` | gpt-4o-mini | ~$0.0005 per question |
| Full gap analysis (72) | gpt-4o-mini | ~$0.03; cached re-run ~$0.0001 |

Gap findings are cached per subcategory in `data/gap_cache.json`, keyed by everything the model saw;
they are re-assessed only when the evidence, mapping, model or prompt changes (`"refresh": true`
forces it).

## Configuration (`.env`)

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | yes | | Embeddings and answers |
| `PINECONE_API_KEY` | yes | | Vector store |
| `PINECONE_INDEX_NAME` | yes | | Index (1536 dims, cosine) |
| `PINECONE_NAMESPACE_STANDARDS` | no | `project-1-standards` | NIST chunks |
| `PINECONE_NAMESPACE_ORG` | no | `project-1-org-default` | Organization chunks + crosswalk |
| `API_KEY` | on a server | (open) | Required `X-API-Key` value |
| `ROUTER_MODEL` | no | `gpt-4o` | Agent for `/ask` |
| `GAP_MODEL` | no | `gpt-4o-mini` | Gap analysis |

## Deploy to Render

Create a **new** Web Service from this repository (don't change any existing service):

| Setting | Value |
|---|---|
| Root Directory | `ai-engineering-bootcamp-v2/Project-1` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| Environment | `OPENAI_API_KEY`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `PINECONE_NAMESPACE_STANDARDS`, `PINECONE_NAMESPACE_ORG`, **`API_KEY`**, `PYTHON_VERSION=3.13.7` (tested; 3.14 also works) |

The server reads the corpus from Pinecone, so ingest from your machine (above), not on Render.
Check `https://<service>.onrender.com/health/pinecone`: `ok: true`, both namespaces populated,
`auth_enabled: true`. Render's free disk is temporary, so the gap cache starts empty after each
deploy or restart; the first `/gaps` run rebuilds it (~$0.03).

## Project layout

```
main.py                  entry point (uvicorn main:app)
app/api.py               routes            app/auth.py        X-API-Key check
app/qa.py                /ask/simple flow  app/agents/        router + tools (/ask)
app/search.py            both-corpus retrieval with labels
app/evidence.py          evidence finder   app/gap_analysis.py  gap engine + cache
app/crosswalk.py         control mapping   app/frameworks.py  subcategory JSON loader
app/prompts.py           answer prompt     app/schemas.py     request/response models
app/retrieval.py         Pinecone + embeddings
app/ingestion/           PDF reader, NIST splitter, internal splitter, Control Mapping parser
ingest_docs.py           corpus -> Pinecone      create_pinecone_index.py  one-time index
ui.py                    Streamlit UI            frameworks/  corpus/  evals/  tests/
```
