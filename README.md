# INTB-groundx-rag
What
Simple RAG demo using GroundX (vector search) + kvant / LiteLLM (OpenAI-compatible /v1/responses). Includes: ingest.py (crawl & chunk), chat.py (CLI RAG), app.py (Streamlit UI).

#Prereqs

Python 3.10+

pip install -r requirements.txt

.env with keys (see below)

.env (required)

GROUNDX_API_KEY=...
GROUNDX_BUCKET_ID=23592
OPENAI_API_KEY=...
OPENAI_API_BASE=https://maas.ai-2.kvant.cloud
OPENAI_MODEL_NAME=inference-llama4-maverick


# Quick commands

Crawl & chunk: python ingest.py → writes out/itnb_chunks_bfs.jsonl

Ingest into GroundX (use GroundX SDK / console) → set GROUNDX_BUCKET_ID

CLI chat: python chat.py

Streamlit UI: streamlit run app.py

# Notes / troubleshooting

If model rejected, script auto-lists /v1/models and retries with available model.

If GroundX search 401 — check GROUNDX_API_KEY and restart process.

Streamlit: upgrade if you hit experimental_rerun errors: pip install --upgrade streamlit.


Keys: never commit API keys. Use .env (local dev) or environment/secrets in deployment.

Data: the crawler scrapes public https://itnb.ch/en only. Make sure you’re allowed to crawl & store the content; comply with site terms.

Logging: sensitive tokens are never logged by the scripts; if you add logs, avoid printing secrets.
