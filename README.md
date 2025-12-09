# INTB-groundx-rag
A compact, production-minded RAG (Retrieval-Augmented Generation) demo that ties together:

GroundX — vector database / search for your ingested site content, and

kvant / LiteLLM proxy (OpenAI-compatible Responses API) — model serving (e.g. inference-llama4-maverick).

This repo contains:

ingest.py — crawler that scrapes only https://itnb.ch/en (no /de or external domains), cleans pages, and writes chunked JSONL ready for GroundX ingestion.

chat.py — headless CLI RAG chat that:

queries GroundX for top-k snippets,

builds a compact RAG prompt,

calls kvant/LiteLLM /v1/responses,

prints the answer and sources.

app.py — Streamlit web UI: chat view, shows sources & debug info; uses st.session_state safely.

requirements.txt — minimal Python deps.

out/ — expected output location (JSONL, debug files).

helper debug files created at runtime (e.g. debug_response_*.json).

Quick start (5–10 minutes)

Create & activate virtualenv:

python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows PowerShell
venv\Scripts\Activate.ps1


Install dependencies:

pip install -r requirements.txt


Create a .env in project root (example):

# GroundX
GROUNDX_API_KEY=your_groundx_api_key
GROUNDX_API_BASE=https://api.groundx.ai         # optional
GROUNDX_BUCKET_ID=23592                         # int; set after ingest or from your GroundX UI

# kvant / LiteLLM (OpenAI-compatible proxy)
OPENAI_API_KEY=sk-...
OPENAI_API_BASE=https://maas.ai-2.kvant.cloud
OPENAI_MODEL_NAME=inference-llama4-maverick

# Optional tuning
RAG_TOP_K=10
RAG_SNIPPET_CHARS=800


Keep your API keys secret. For local development .env is fine; in production use a secrets manager.

Workflow
1) Crawl & prepare chunks

Run the crawler which writes out/itnb_chunks_bfs.jsonl and out/itnb_contact_info_aggregated.json:

python ingest.py


Output: chunked JSON lines — one JSON object per chunk:

{ "url":"https://itnb.ch/en/...", "chunk_id":"...", "text":"...", "file_name":"..._chunk-0.txt" }


Use this JSONL to upload into GroundX (via SDK or HTTP API).

2) Ingest into GroundX

You can ingest with the official GroundX SDK (preferred) or via their HTTP/CLI. Example (pseudo):

from groundx import GroundX, Document
g = GroundX(api_key="...")
bucket = g.buckets.create(name="itnb-rag-bucket")
# iterate over JSONL and upload documents...


Or use a small upload script to send the files to GroundX following GroundX docs.

Note: set GROUNDX_BUCKET_ID in .env to the bucket id you created.

3) CLI RAG chat

Run:

python chat.py


Type questions (e.g. Tell me about the Sovereign Orchestrator) and the script will:

search GroundX (top-k),

build a prompt with retrieved snippets,

call /v1/responses on your kvant proxy,

print the model reply and list the sources.

Recommended test prompt:

Explain the key capabilities of the Sovereign Orchestrator and how it ensures workflow security. Cite exact sections from retrieved content.

4) Web UI (Streamlit)

Run:

streamlit run app.py


Open the local URL printed by Streamlit. The UI shows:

chat history,

sources used,

debug controls (clear chat),

debug JSON files are saved to repo root for inspection.

Important configuration notes & troubleshooting
GroundX search failing (401 / "You did not include your API key")

Ensure GROUNDX_API_KEY is set in .env and you restarted the process after updating .env.

Test simple call:

python -c "from groundx import GroundX; print(GroundX(api_key='YOUR_KEY').buckets.list())"

Model call fails with Invalid model name

Your kvant proxy may not expose openai/inference-llama4-maverick under that exact name.

Use curl or script to list models:

curl -H "Authorization: Bearer $OPENAI_API_KEY" https://maas.ai-2.kvant.cloud/v1/models


chat.py and app.py will attempt to auto-fallback by fetching /v1/models and trying a first available model if the preferred model is rejected.

Streamlit errors

If you see AttributeError: module 'streamlit' has no attribute 'experimental_rerun' — upgrade Streamlit to a modern version (pip install --upgrade streamlit). The code uses st.experimental_rerun() which is available in commonly used Streamlit versions. If your version is old, upgrade.

If you see st.session_state... cannot be modified after the widget with key ... is instantiated — that occurs if code assigns st.session_state keys after creating widgets with the same keys. The provided app.py initializes st.session_state only when keys are missing to avoid this. If you changed the code, follow the pattern:

if "query_input" not in st.session_state:
    st.session_state.query_input = ""

GroundX SDK vs HTTP shapes

Different GroundX SDK / server versions return slightly different JSON shapes. The scripts are defensive and attempt to read both resp.search.results, results, and data forms.

Prompt guidance & safety posture

The RAG prompt builder instructs the model to answer only using provided snippets and to cite sources inline. Good practice:

Keep TOP_K 6–10 for focused retrieval.

Trim snippet length via RAG_SNIPPET_CHARS to avoid overly long prompts.

If the model hallucinate or invents, add explicit guardrails or increase citation requirements.

Example small prompt snippet produced by the code:

You are an assistant that answers user questions ONLY using the provided source snippets...
User question: Explain X

Retrieved snippets (top 3):
[1] Source: https://itnb.ch/en/products/.../sovereign-orchestrator
...text...

Answer concisely and cite sources using the [source] format.

File overview

ingest.py — crawler + chunker (writes out/itnb_chunks_bfs.jsonl)

chat.py — CLI RAG program

app.py — Streamlit chat UI

requirements.txt — Python dependencies

.env.example — example environment variables (create .env from this)

README.md — this file

Security & privacy

Keys: never commit API keys. Use .env (local dev) or environment/secrets in deployment.

Data: the crawler scrapes public https://itnb.ch/en only. Make sure you’re allowed to crawl & store the content; comply with site terms.

Logging: sensitive tokens are never logged by the scripts; if you add logs, avoid printing secrets.
