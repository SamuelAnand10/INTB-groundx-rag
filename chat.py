#!/usr/bin/env python3
"""
chat.py - Headless RAG chat using GroundX (vector search) + kvant/LiteLLM Responses API.

Usage:
  python chat.py

Environment variables (use .env + python-dotenv or export in shell):
  GROUNDX_API_KEY        - GroundX API key
  GROUNDX_API_BASE       - optional GroundX base (default https://api.groundx.ai)
  GROUNDX_BUCKET_ID      - bucket id (int) with your ingested documents
  OPENAI_API_KEY         - kvant / LiteLLM proxy API key
  OPENAI_API_BASE        - e.g. https://maas.ai-2.kvant.cloud
  OPENAI_MODEL_NAME      - optional model (recommended: inference-llama4-maverick)

This script:
 - looks up top-k hits from GroundX,
 - constructs a short prompt containing the top snippets and source urls,
 - calls the /v1/responses endpoint on your kvant proxy with that prompt,
 - prints the model reply and the sources used.
"""
import os
import time
import json
import logging
from typing import Dict, Any, List, Optional

import requests
from dotenv import load_dotenv

# optional GroundX SDK - used if available; otherwise we fallback to manual HTTP
try:
    from groundx import GroundX
    HAVE_GROUNDX_SDK = True
except Exception:
    HAVE_GROUNDX_SDK = False

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------- CONFIG (from env) ----------
GROUNDX_API_KEY = os.getenv("GROUNDX_API_KEY")
GROUNDX_API_BASE = os.getenv("GROUNDX_API_BASE", "https://api.groundx.ai").rstrip("/")
GROUNDX_BUCKET_ID = os.getenv("GROUNDX_BUCKET_ID")  # must be set to your bucket id (int)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://maas.ai-2.kvant.cloud").rstrip("/")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "inference-llama4-maverick")
TOP_K = int(os.getenv("RAG_TOP_K", "10"))
MAX_CHARS_FROM_SNIPPET = int(os.getenv("RAG_SNIPPET_CHARS", "800"))
# ---------------------------------------

# Basic checks
if not GROUNDX_API_KEY:
    logging.warning("GROUNDX_API_KEY not set. GroundX search will fail unless you set it.")
if not GROUNDX_BUCKET_ID:
    logging.error("GROUNDX_BUCKET_ID not set. Please set it to the bucket id you created when ingesting.")
    # Do not exit - allow user to still run non-search flows, but most use cases will need this.
if not OPENAI_API_KEY:
    logging.error("OPENAI_API_KEY not set. Please set kvant proxy key in env OPENAI_API_KEY.")
    raise SystemExit(1)


# ---------------- GroundX search ----------------

def search_groundx(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Query GroundX and return a list of hits (dictionaries).
    Each hit will contain at least: 'text' and 'source_url' (if available), 'file_name'.
    Uses GroundX SDK if available; otherwise calls HTTP endpoint (/api/v1/search/content).
    """
    hits = []
    if not GROUNDX_BUCKET_ID:
        logging.error("No GROUNDX_BUCKET_ID configured; cannot search.")
        return hits

    # Try SDK first
    try:
        gx = GroundX(api_key=GROUNDX_API_KEY)

    except Exception as e:
        logging.warning("GroundX SDK present but failed to instantiate: %s", e)

    try:
        if gx:
            logging.info("Searching GroundX (SDK) for: %s", query)
            # SDK shape: groundx.search.content(id=BUCKET_ID, query=...)
            resp = gx.search.content(id=23592,query=query, n=top_k)
            # resp.search.results is list of SearchResultItem objects
            results = getattr(resp.search, "results", []) if hasattr(resp, "search") else getattr(resp, "results", [])
            for r in results[:top_k]:
                hits.append({
                    "text": getattr(r, "text", None) or getattr(r, "suggested_text", None) or "",
                    "source_url": getattr(r, "source_url", None) or getattr(r, "multimodal_url", None) or None,
                    "file_name": getattr(r, "file_name", None),
                    "score": getattr(r, "score", None),
                })
            return hits
    except Exception as e:
        logging.warning("GroundX SDK search failed: %s - falling back to HTTP", e)

    # Fallback: HTTP call to GroundX API
    try:
        url = GROUNDX_API_BASE + "/api/v1/search/content"
        headers = {"Authorization": f"Bearer {GROUNDX_API_KEY}", "Content-Type": "application/json"}
        payload = {"id": int(GROUNDX_BUCKET_ID), "query": query, "top_k": top_k}
        logging.info("Searching GroundX (HTTP) %s ...", url)
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        jr = r.json()
        # Defensive parsing - look for jr['search']['results'] etc.
        results = jr.get("search", {}).get("results") or jr.get("results") or jr.get("data") or []
        for item in results[:top_k]:
            # item may be dict-like
            hits.append({
                "text": item.get("text") or item.get("suggested_text") or "",
                "source_url": item.get("source_url") or item.get("multimodal_url") or None,
                "file_name": item.get("file_name"),
                "score": item.get("score"),
            })
        return hits
    except Exception as e:
        logging.error("GroundX HTTP search failed: %s", e)
        return hits


# --------------- kvant / LiteLLM / Responses API helpers ---------------
def list_available_models() -> List[str]:
    """Call /v1/models and return a list of model ids (defensive)."""
    url = OPENAI_API_BASE + "/v1/models"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        j = r.json()
        models = []
        if isinstance(j, dict) and "data" in j and isinstance(j["data"], list):
            for m in j["data"]:
                if isinstance(m, dict):
                    mid = m.get("id") or m.get("model") or m.get("name")
                    if mid:
                        models.append(mid)
        elif isinstance(j, list):
            for m in j:
                if isinstance(m, dict):
                    mid = m.get("id") or m.get("name")
                    if mid:
                        models.append(mid)
        # fallback: collect strings from JSON
        if not models:
            def collect(o, out):
                if len(out) > 200: return
                if isinstance(o, str):
                    out.append(o)
                elif isinstance(o, dict):
                    for v in o.values():
                        collect(v, out)
                elif isinstance(o, list):
                    for v in o:
                        collect(v, out)
            temp = []
            collect(j, temp)
            # filter likely model ids (no spaces, not too long)
            models = [s for s in temp if isinstance(s, str) and " " not in s][:200]
        return models
    except Exception as e:
        logging.error("Failed to list models from kvant proxy: %s", e)
        return []


def call_openai_model_responses_auto(prompt: str, model: Optional[str] = None,
                                    max_output_tokens: int = 768) -> Dict[str, Any]:
    """
    Call /v1/responses on kvant proxy robustly.
    If model not given, use OPENAI_MODEL_NAME. If the server rejects model, fetch /v1/models and retry.
    Returns JSON response (raises on HTTP errors).
    """
    model = model or OPENAI_MODEL_NAME
    url = OPENAI_API_BASE.rstrip("/") + "/v1/responses"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "input": prompt, "max_output_tokens": max_output_tokens}

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code == 400 and ("Invalid model" in r.text or "Invalid model name" in r.text):
        logging.warning("Model '%s' invalid per server. Querying available models...", model)
        models = list_available_models()
        if not models:
            logging.error("No models returned by /v1/models; cannot recover.")
            r.raise_for_status()
        # pick first reasonable alternative
        alt = models[0]
        logging.info("Retrying with model: %s", alt)
        payload["model"] = alt
        time.sleep(0.25)
        r2 = requests.post(url, headers=headers, json=payload, timeout=60)
        r2.raise_for_status()
        return r2.json()
    # else normal path
    r.raise_for_status()
    return r.json()


def extract_text_from_responses(resp: Dict[str, Any]) -> str:
    """
    Defensive extraction of text from a Responses API JSON result (see earlier).
    """
    if not resp:
        return ""
    if isinstance(resp.get("output_text"), str) and resp["output_text"].strip():
        return resp["output_text"].strip()
    out = resp.get("output") or resp.get("outputs")
    if isinstance(out, list) and out:
        pieces = []
        for item in out:
            if not isinstance(item, dict): continue
            content = item.get("content") or item.get("contents") or item.get("data") or []
            if isinstance(content, list):
                for c in content:
                    if not isinstance(c, dict): continue
                    if "text" in c and isinstance(c["text"], str):
                        pieces.append(c["text"])
                    elif c.get("type") in ("output_text", "message") and isinstance(c.get("text"), str):
                        pieces.append(c.get("text"))
                    elif "parts" in c and isinstance(c["parts"], list):
                        pieces.append("".join([p for p in c["parts"] if isinstance(p, str)]))
                    elif "value" in c and isinstance(c["value"], str):
                        pieces.append(c["value"])
        if pieces:
            return "\n".join(pieces).strip()

    choices = resp.get("choices")
    if isinstance(choices, list) and choices:
        texts = []
        for c in choices:
            msg = c.get("message")
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str):
                    texts.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and "text" in part:
                            texts.append(part["text"])
                        elif isinstance(part, str):
                            texts.append(part)
            if "text" in c and isinstance(c["text"], str):
                texts.append(c["text"])
        if texts:
            return "\n".join(texts).strip()

    gens = resp.get("generations")
    if isinstance(gens, list) and gens:
        g0 = gens[0]
        if isinstance(g0, dict) and isinstance(g0.get("text"), str):
            return g0["text"].strip()

    # fallback: find first string deep in JSON
    def find_first_string(obj):
        if isinstance(obj, str): return obj
        if isinstance(obj, dict):
            for v in obj.values():
                s = find_first_string(v)
                if s: return s
        if isinstance(obj, list):
            for v in obj:
                s = find_first_string(v)
                if s: return s
        return None
    fb = find_first_string(resp)
    return fb.strip() if fb else ""


# ---------------- prompt builder ----------------
SYSTEM_INSTRUCTION = (
    "You are an assistant that answers user questions ONLY using the provided source snippets. "
    "If the answer is not present in the snippets, say you don't know and optionally suggest where to look. "
    "Cite sources inline by [source_url] after the sentence when appropriate."
)


def build_rag_prompt(user_query: str, hits: List[Dict[str, Any]]) -> str:
    """
    Build a compact RAG prompt: system instruction, user question, and top-K snippets with their sources.
    """
    prompt_parts = []
    prompt_parts.append(SYSTEM_INSTRUCTION)
    prompt_parts.append("\nUser question:\n" + user_query.strip() + "\n")
    if not hits:
        prompt_parts.append("\n(No retrieved documents; answer from general knowledge only.)\n")
    else:
        prompt_parts.append("Retrieved snippets (top {}):\n".format(len(hits)))
        for i, h in enumerate(hits, start=1):
            text = (h.get("text") or "").strip()
            if len(text) > MAX_CHARS_FROM_SNIPPET:
                text = text[:MAX_CHARS_FROM_SNIPPET] + " ... (truncated)"
            src = h.get("source_url") or h.get("file_name") or "source:unknown"
            prompt_parts.append(f"[{i}] Source: {src}\n{ text }\n")
    prompt_parts.append("\nAnswer concisely and cite sources using the [source] format where helpful.\n")
    return "\n".join(prompt_parts)


# ---------------- CLI chat loop ----------------
def cli_chat_loop():
    print("Headless RAG chat (GroundX + kvant). Type 'exit' to quit.")
    while True:
        q = input("\nQ: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            print("bye")
            break

        # 1) retrieve
        hits = search_groundx(q, top_k=TOP_K)
        if hits:
            print(f"Retrieved {len(hits)} hits from GroundX (showing top {TOP_K}).")
            for i, h in enumerate(hits, start=1):
                src = h.get("source_url") or h.get("file_name")
                score = h.get("score")
                print(f"  [{i}] src={src} score={score}")
        else:
            print("No hits from GroundX; continuing with model-only prompt.")

        # 2) build RAG prompt
        prompt = build_rag_prompt(q, hits)

        # 3) call model
        try:
            resp_json = call_openai_model_responses_auto(prompt, model=OPENAI_MODEL_NAME)
        except Exception as e:
            logging.error("Model call failed: %s", e)
            print("Model call failed; see logs.")
            continue

        # 4) extract answer
        answer_text = extract_text_from_responses(resp_json)
        print("\n=== Model answer ===\n")
        if answer_text:
            print(answer_text)
        else:
            print("(no text extracted from model response; raw JSON below)")
            print(json.dumps(resp_json, indent=2)[:4000])

        # 5) show used sources for user
        if hits:
            print("\nSources:")
            for i, h in enumerate(hits, start=1):
                print(f" [{i}] {h.get('source_url') or h.get('file_name')}")

        # optional: save raw response to debug file
        try:
            timestamp = int(time.time())
            fname = f"debug_response_{timestamp}.json"
            with open(fname, "w", encoding="utf-8") as f:
                json.dump({"query": q, "groundx_hits": hits, "resp_json": resp_json}, f, ensure_ascii=False, indent=2)
            logging.info("Saved raw response to %s", fname)
        except Exception:
            pass


if __name__ == "__main__":
    # quick startup info
    logging.info("GROUNDX_API_KEY present?: %s", bool(GROUNDX_API_KEY))
    logging.info("GROUNDX_BUCKET_ID present?: %s", bool(GROUNDX_BUCKET_ID))
    logging.info("OPENAI_API_BASE: %s", OPENAI_API_BASE)
    logging.info("OPENAI_MODEL_NAME (preferred): %s", OPENAI_MODEL_NAME)
    cli_chat_loop()
