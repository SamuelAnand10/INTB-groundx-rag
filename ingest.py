# ingest.py
import os
import json
import time
import shutil
from pathlib import Path
from typing import List
import logging
from dotenv import load_dotenv
load_dotenv()


# Install the GroundX SDK per docs, e.g. pip install groundx
# The docs show an SDK with classes Document, GroundX:
# from groundx import Document, GroundX
try:
    from groundx import Document, GroundX
except Exception as e:
    raise RuntimeError("Install the official groundx SDK (pip install groundx) and ensure import works.") from e

# ---------- CONFIG ----------
API_KEY = os.getenv("GROUNDX_API_KEY") or "YOUR_GROUNDX_API_KEY"
INPUT_JSONL = "out/itnb_chunks_bfs.jsonl"
TMP_DIR = Path("tmp_ingest")
BATCH_SIZE = 20    # GroundX docs recommend batch_size 1..50 (default 10). tune as desired.
WAIT_FOR_COMPLETE = True  # block until GroundX finishes processing each batch
RETRY_ATTEMPTS = 2
RETRY_DELAY = 3  # seconds
BUCKET_ID = 23592
# ----------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")



def prepare_tmp_files(input_jsonl: str, tmp_dir: Path) -> List[Path]:
    """Read JSONL and write per-chunk local text files. Returns list of file paths to ingest."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    file_paths = []
    with open(input_jsonl, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            try:
                doc = json.loads(line)
            except Exception:
                logging.exception("Skipping invalid JSON line %d", i)
                continue
            # build safe filename
            doc_id = doc.get("id", f"chunk-{i}")
            # sanitize filename
            safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in doc_id)
            fname = tmp_dir / f"{safe_name}.txt"
            with open(fname, "w", encoding="utf-8") as fo:
                # write useful metadata header for debugging, then chunk content
                fo.write(f"URL: {doc.get('url','')}\n")
                fo.write(f"Title: {doc.get('title','')}\n\n")
                fo.write(doc.get("text", ""))
            file_paths.append(fname)
    logging.info("Prepared %d local files for ingest in %s", len(file_paths), tmp_dir)
    return file_paths

def make_document_obj(local_path: str, file_name: str, bucket_id: int):
    """
    Construct a Document object expected by GroundX SDK.
    Docs example uses: Document(bucket_id=..., file_name="...", file_path="/local/path/file2.pdf", file_type="pdf")
    We'll use file_type "txt".
    """
    return Document(
        bucket_id=bucket_id,
        file_name=file_name,
        file_path=str(local_path),
        file_type="txt",
    )

def ingest_batch(client, documents: List[Document], batch_size: int = 20, wait=False):
    """
    Call client.ingest for a list of Document objects.
    Returns client response object (SDK-defined). Handles retries.
    """
    for attempt in range(RETRY_ATTEMPTS + 1):
        try:
            resp = client.ingest(documents=documents, wait_for_complete=wait, batch_size=len(documents))
            return resp
        except Exception as e:
            logging.exception("Ingest attempt %d failed for batch: %s", attempt+1, e)
            if attempt < RETRY_ATTEMPTS:
                logging.info("Retrying in %s seconds...", RETRY_DELAY)
                time.sleep(RETRY_DELAY)
            else:
                raise

def main():
    if API_KEY == "YOUR_GROUNDX_API_KEY" or not API_KEY:
        logging.error("Set your GroundX API key in GROUNDX_API_KEY or edit API_KEY in this script.")
        return
    if BUCKET_ID == 1234:
        logging.warning("BUCKET_ID is still the placeholder 1234 — set your real bucket id before ingesting.")

    # init client
    client = GroundX(api_key=API_KEY)

    # prepare files from jsonl
    files = prepare_tmp_files(INPUT_JSONL, TMP_DIR)
    if not files:
        logging.error("No files prepared for ingest — check %s", INPUT_JSONL)
        return

    # chunk into batches and call client.ingest
    total = len(files)
    logging.info("Starting ingest: %d files, batch_size=%d", total, BATCH_SIZE)
    for i in range(0, total, BATCH_SIZE):
        batch_files = files[i:i+BATCH_SIZE]
        docs = []
        for fp in batch_files:
            docs.append(make_document_obj(local_path=fp, file_name=fp.name, bucket_id=BUCKET_ID))
        logging.info("Ingesting batch %d..%d (size=%d)", i+1, min(i+BATCH_SIZE, total), len(docs))
        try:
            resp = ingest_batch(client, docs, batch_size=len(docs), wait=WAIT_FOR_COMPLETE)
            logging.info("Batch ingest response: %s", getattr(resp, "status", resp))
        except Exception as e:
            logging.exception("Failed to ingest batch starting at %d: %s", i+1, e)

    logging.info("Ingest finished. You can now call GroundX search endpoints to query your documents.")
    # optionally cleanup temporary files
    # shutil.rmtree(TMP_DIR)
    # logging.info("Removed temporary files at %s", TMP_DIR)

if __name__ == "__main__":
    main()
