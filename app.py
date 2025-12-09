# app.py
import os
import time
import json
import logging
from typing import Dict, Any, List, Optional

import requests
import streamlit as st
from dotenv import load_dotenv

# optional GroundX SDK - used if available; otherwise we fallback to manual HTTP
try:
    from groundx import GroundX
    HAVE_GROUNDX_SDK = True
except Exception:
    HAVE_GROUNDX_SDK = False

# load .env if present
load_dotenv()

# logging
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

# Streamlit page config
st.set_page_config(page_title="GroundX + kvant RAG Chat", layout="wide")

# ----------------- session state initialization (very important) -----------------
# create keys BEFORE any widgets are instantiated to avoid Streamlit errors
if "query_input" not in st.session_state:
    st.session_state["query_input"] = ""
if "history" not in st.session_state:
    # history: list of dicts: {role: "user"|"assistant", text: str, sources: List[str], debug_fname: str}
    st.session_state["history"] = []
if "last_debug" not in st.session_state:
    st.session_state["last_debug"] = None
if "clear_next" not in st.session_state:
    st.session_state["clear_next"] = False
if "loading" not in st.session_state:
    st.session_state["loading"] = False

# ---------------- GroundX search ----------------
def search_groundx(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    if not GROUNDX_BUCKET_ID:
        logging.error("No GROUNDX_BUCKET_ID configured; cannot search.")
        return hits

    # Try SDK first (if available)
    gx = None
    if HAVE_GROUNDX_SDK:
        try:
            gx = GroundX(api_key=GROUNDX_API_KEY)
        except Exception as e:
            logging.warning("GroundX SDK present but failed to instantiate: %s", e)
            gx = None

    try:
        if gx:
            logging.info("Searching GroundX (SDK) for: %s", query)
            resp = gx.search.content(id=int(GROUNDX_BUCKET_ID), query=query, n=top_k)
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

    # Fallback: HTTP call
    try:
        url = GROUNDX_API_BASE + "/api/v1/search/content"
        headers = {"Authorization": f"Bearer {GROUNDX_API_KEY}", "Content-Type": "application/json"}
        payload = {"id": int(GROUNDX_BUCKET_ID), "query": query, "top_k": top_k}
        logging.info("Searching GroundX (HTTP) %s ...", url)
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        jr = r.json()
        results = jr.get("search", {}).get("results") or jr.get("results") or jr.get("data") or []
        for item in results[:top_k]:
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
        models: List[str] = []
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
            temp: List[str] = []
            collect(j, temp)
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
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")

    model = model or OPENAI_MODEL_NAME
    url = OPENAI_API_BASE.rstrip("/") + "/v1/responses"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "input": prompt, "max_output_tokens": max_output_tokens}

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    # Detect invalid model error text (some proxies return 400 with message)
    if r.status_code == 400 and ("Invalid model" in r.text or "Invalid model name" in r.text or "Invalid model name passed" in r.text):
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
    r.raise_for_status()
    return r.json()

def extract_text_from_responses(resp: Dict[str, Any]) -> str:
    """Defensive extraction of text from a Responses API JSON result."""
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

    # fallback: find first string in JSON
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

# ----------------- UI -----------------
st.title("GroundX + kvant RAG Chat")
st.markdown("Ask questions — the app will search your GroundX bucket and call the kvant/LiteLLM proxy to answer.")

# layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Chat")
    # Clear input on the next run if flagged (do it before creating the widget)
    if st.session_state.get("clear_next", False):
        st.session_state["query_input"] = ""
        st.session_state["clear_next"] = False

    # Use a form so submit is atomic and reduces reruns
    with st.form(key="query_form", clear_on_submit=False):
        user_q = st.text_area("Your question", value=st.session_state["query_input"], key="query_input", height=140)
        submit_btn = st.form_submit_button("Send")
        clear_btn = st.form_submit_button("Clear chat", on_click=None)

    # If clear button pressed: schedule clearing
    if clear_btn:
        st.session_state["history"] = []
        st.session_state["last_debug"] = None
        st.session_state["clear_next"] = True
        st.success("Chat cleared.")
        # skip processing send logic this run
    elif submit_btn and user_q.strip():
        query = user_q.strip()
        st.session_state["loading"] = True
        with st.spinner("Searching and querying model..."):
            # 1) retrieve
            hits = []
            if GROUNDX_API_KEY and GROUNDX_BUCKET_ID:
                try:
                    hits = search_groundx(query, top_k=TOP_K)
                except Exception as e:
                    logging.error("Search failed: %s", e)

            # 2) build prompt
            prompt = build_rag_prompt(query, hits)

            # 3) call model
            try:
                resp_json = call_openai_model_responses_auto(prompt, model=OPENAI_MODEL_NAME)
            except Exception as e:
                logging.error("Model call failed: %s", e)
                st.error(f"Model call failed: {e}")
                resp_json = {}

            answer_text = extract_text_from_responses(resp_json) or "(no text returned)"
            # save debug file
            timestamp = int(time.time())
            fname = f"debug_response_{timestamp}.json"
            try:
                with open(fname, "w", encoding="utf-8") as f:
                    json.dump({"query": query, "groundx_hits": hits, "resp_json": resp_json}, f, ensure_ascii=False, indent=2)
                st.session_state["last_debug"] = fname
            except Exception as e:
                logging.warning("Failed to save debug file: %s", e)
                st.session_state["last_debug"] = None

            # 4) append to history
            st.session_state["history"].append({
                "role": "user",
                "text": query,
                "sources": [h.get("source_url") or h.get("file_name") for h in hits],
                "debug": st.session_state["last_debug"],
            })
            st.session_state["history"].append({
                "role": "assistant",
                "text": answer_text,
                "sources": [h.get("source_url") or h.get("file_name") for h in hits],
                "debug": st.session_state["last_debug"],
            })

            # schedule clearing input on next run (safe)
            st.session_state["clear_next"] = True
            st.session_state["loading"] = False

# show chat history (most recent at bottom)
with col1:
    st.subheader("Conversation")
    if not st.session_state["history"]:
        st.info("No messages yet — ask a question to begin.")
    else:
        # show each message
        for i, msg in enumerate(st.session_state["history"]):
            role = msg.get("role", "user")
            text = msg.get("text", "")
            if role == "user":
                st.markdown(f"**You:** {text}")
            else:
                st.markdown(f"**Assistant:** {text}")
            # show condensed sources (if any)
            sources = msg.get("sources", []) or []
            if sources:
                with st.expander("Sources", expanded=False):
                    for s in sources:
                        if s:
                            # try to show clickable link if it's a http(s) url
                            if isinstance(s, str) and s.startswith("http"):
                                st.write(f"- [{s}]({s})")
                            else:
                                st.write(f"- {s}")
            # show debug file link if available
            dbg = msg.get("debug")
            if dbg:
                st.write(f"_Debug saved:_ `{dbg}`")
            st.markdown("---")

with col2:
    st.subheader("Top-k retrieved")
    last_query_sources = []
    if st.session_state["history"]:
        # pick last assistant entry sources
        last_assistant = None
        for m in reversed(st.session_state["history"]):
            if m.get("role") == "assistant":
                last_assistant = m
                break
        if last_assistant:
            last_query_sources = last_assistant.get("sources", []) or []
    if last_query_sources:
        for idx, s in enumerate(last_query_sources, start=1):
            if s and isinstance(s, str) and s.startswith("http"):
                st.markdown(f"{idx}. [{s}]({s})")
            else:
                st.markdown(f"{idx}. {s}")
    else:
        st.write("No retrieved sources to show (yet).")

    st.markdown("---")
    st.subheader("Settings & status")
    st.write(f"GROUNDX configured: {bool(GROUNDX_API_KEY and GROUNDX_BUCKET_ID)}")
    st.write(f"OPENAI proxy configured: {bool(OPENAI_API_KEY)}")
    st.write(f"Preferred model: `{OPENAI_MODEL_NAME}`")
    if st.session_state.get("last_debug"):
        st.write("Last debug file:", st.session_state["last_debug"])

    st.markdown("**Note:** Clearing input is scheduled to happen *before* the next run (safe).")

# end of file
