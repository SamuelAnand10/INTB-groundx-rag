#!/usr/bin/env python3
"""
crawl_bfs_itnb_prefix_only.py

Same behavior as earlier crawler, but STRICTLY only allows URLs that start with ALLOWED_PREFIX:
    ALLOWED_PREFIX = "https://itnb.ch/en"

This prevents crawling images and other domains or locales such as /de or external links.
"""
import asyncio
import json
import os
import re
from urllib.parse import urlparse, urljoin

from crawl4ai import AsyncWebCrawler  # requires crawl4ai installed

# -------- CONFIG --------
SEED_URL = "https://itnb.ch/en"
ALLOWED_PREFIX = "https://itnb.ch/en"   # <-- Only URLs starting with this will be crawled/queued
OUT_DIR = "out"
RAW_OUT = os.path.join(OUT_DIR, "itnb_raw_bfs.jsonl")
CHUNKS_OUT = os.path.join(OUT_DIR, "itnb_chunks_bfs.jsonl")

MAX_DEPTH = 2
MAX_PAGES_TOTAL = 200
RATE_DELAY = 0.4
CHUNK_MAX_CHARS = 1600

os.makedirs(OUT_DIR, exist_ok=True)

# -------- Regexes & helpers --------
_contact_email_re = re.compile(r'([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})')
_contact_phone_re = re.compile(r'(\+?\d[\d\s\-/\(\)]{6,}\d)')
IMAGE_URL_RE = re.compile(r'https?://[^\s)]+\.(?:jpe?g|png|gif|bmp|svg|webp|ico)(?:\?[^\s)]*)?', flags=re.I)
_image_url_line_re = IMAGE_URL_RE

def is_image_url(url: str) -> bool:
    return bool(url and IMAGE_URL_RE.search(url))

def is_allowed_url(url: str) -> bool:
    """Only allow urls that start exactly with ALLOWED_PREFIX (normalize fragment drop)."""
    if not url:
        return False
    u = url.split('#')[0]
    return u.startswith(ALLOWED_PREFIX)

def remove_header_nav(md: str) -> str:
    if not md:
        return ""
    m = re.search(r'(^|\n)(#{1,3}\s+.+)', md)
    if m:
        start = m.start(2)
        return md[start:].lstrip()
    for i, line in enumerate(md.splitlines()):
        if len(line.strip()) >= 40:
            return "\n".join(md.splitlines()[i:]).lstrip()
    return md

def remove_image_urls(md: str) -> str:
    if not md:
        return ""
    lines = md.splitlines()
    kept = []
    for ln in lines:
        if re.search(_image_url_line_re, ln):
            if re.fullmatch(r'\s*' + r'(' + _image_url_line_re.pattern + r')\s*', ln, flags=re.I):
                continue
            ln = re.sub(_image_url_line_re, '', ln)
            if not ln.strip():
                continue
        kept.append(ln)
    return "\n".join(kept)

def extract_and_remove_contact(md: str):
    if not md:
        return "", ""
    original = md
    contact_blocks = []
    for m in re.finditer(r'(^|\n)(#{1,3}\s*.*Contact.*|.*\bGet in Contact\b.*|.*\bContact Us\b.*)', md, flags=re.I):
        following = md[m.end():].splitlines()
        block_lines = [m.group(0).strip()]
        for ln in following[:8]:
            if not ln.strip():
                break
            block_lines.append(ln.rstrip())
        blk = "\n".join(block_lines).strip()
        if blk:
            contact_blocks.append((m.start(), m.end(), blk))
    lines = md.splitlines()
    tail = lines[max(0, len(lines)-40):]
    tail_text = "\n".join(tail)
    if ("contact" in tail_text.lower()) or re.search(_contact_email_re, tail_text) or re.search(_contact_phone_re, tail_text):
        match_indices = [i for i, l in enumerate(tail) if re.search(r'contact|@|tel|phone|\+?\d{2,}', l, flags=re.I)]
        if match_indices:
            start_idx = max(0, match_indices[0]-2)
            end_idx = min(len(tail), match_indices[-1]+3)
            trailing_block = "\n".join(tail[start_idx:end_idx]).strip()
            if trailing_block:
                contact_blocks.append(("tail", None, trailing_block))

    emails = set(re.findall(_contact_email_re, md))
    phones = set(re.findall(_contact_phone_re, md))
    contact_links = set(re.findall(r'\[.*?\]\((https?://[^\)]+contact[^\)]*)\)', md, flags=re.I))
    contact_rel_links = set(re.findall(r'\[.*?\]\((/[^)\s]*contact[^)]*)\)', md, flags=re.I))
    for rl in contact_rel_links:
        contact_links.add(urljoin(ALLOWED_PREFIX, rl))

    contact_pieces = []
    for cb in contact_blocks:
        blk = cb[2] if isinstance(cb, tuple) else str(cb)
        blk = re.sub(r'\n{2,}', '\n', blk).strip()
        if blk and blk not in contact_pieces:
            contact_pieces.append(blk)

    for e in sorted(emails):
        if e not in "\n".join(contact_pieces):
            contact_pieces.append(f"Email: {e}")
    for p in sorted(phones):
        if len(re.sub(r'\D', '', p)) >= 7 and p not in "\n".join(contact_pieces):
            contact_pieces.append(f"Phone: {p}")
    # only include contact links that are allowed by the prefix
    for cl in sorted(contact_links):
        if cl.startswith(ALLOWED_PREFIX) and cl not in "\n".join(contact_pieces):
            contact_pieces.append(f"Contact page: {cl}")

    contact_text = "\n\n".join(contact_pieces).strip()

    cleaned_md = original
    for cb in contact_blocks:
        blk = cb[2]
        if blk:
            cleaned_md = cleaned_md.replace(blk, "")

    cleaned_md = re.sub(r'^\s*(Contact|Contact Us|Get in Contact)[^\n]*\n?', '', cleaned_md, flags=re.I | re.M)
    cleaned_md = re.sub(r'^\s*[\w\.-]+@[\w\.-]+\.\w+\s*$', '', cleaned_md, flags=re.M)
    cleaned_md = re.sub(r'^\s*\+?\d[\d\s\-/\(\)]{6,}\d\s*$', '', cleaned_md, flags=re.M)
    cleaned_md = re.sub(r'\n{3,}', '\n\n', cleaned_md).strip()

    return cleaned_md, contact_text

def clean_markdown(md: str):
    if not md:
        return "", ""
    md = remove_header_nav(md)
    md = re.sub(r'!\[.*?\]\(.*?\)', '', md)
    md = re.sub(r'<img [^>]*>', '', md, flags=re.I)
    md = remove_image_urls(md)
    md = re.sub(r'Download\s*\(.*?\)', '', md)
    # remove long nav-like bullet runs
    lines = md.splitlines()
    cleaned_lines = []
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith(("* ", "- ")):
            j = i
            while j < len(lines) and lines[j].strip().startswith(("* ", "- ")):
                j += 1
            if (j - i) > 6:
                i = j
                continue
        cleaned_lines.append(lines[i])
        i += 1
    md = "\n".join(cleaned_lines)
    md, contact_snippet = extract_and_remove_contact(md)
    md = re.sub(r'\n{3,}', '\n\n', md).strip()
    return md, contact_snippet

def chunk_text(text: str, max_chars: int = CHUNK_MAX_CHARS):
    if not text:
        return []
    parts = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        br = text.rfind("\n\n", start, end)
        if br <= start:
            br = text.rfind(". ", start, end)
        if br <= start:
            br = end
        chunk = text[start:br].strip()
        chunk = re.sub(IMAGE_URL_RE, '', chunk)
        chunk = re.sub(r'<img [^>]*>', '', chunk, flags=re.I)
        if chunk and len(chunk) > 40:
            parts.append(chunk)
        start = br
    return parts

def extract_internal_links(text: str, allowed_prefix: str = ALLOWED_PREFIX):
    found = set()
    # markdown links first
    for m in re.finditer(r'\[.*?\]\((https?://[^\)]+)\)', text):
        u = m.group(1)
        # skip images and non-allowed prefixes
        if is_image_url(u):
            continue
        if u.startswith(allowed_prefix):
            found.add(u.split('#')[0])
    # bare links
    for m in re.finditer(r'https?://[A-Za-z0-9\-\._~:/?#\[\]@!$&\'()*+,;=%]+', text):
        u = m.group(0)
        if is_image_url(u):
            continue
        if u.startswith(allowed_prefix):
            found.add(u.split('#')[0])
    return found

async def bfs_crawl(seed_url: str = SEED_URL,
                    max_depth: int = MAX_DEPTH,
                    max_pages_total: int = MAX_PAGES_TOTAL,
                    rate_delay: float = RATE_DELAY):
    seen = set()
    results = []
    queue = [(seed_url, 0)]
    async with AsyncWebCrawler() as crawler:
        while queue and len(results) < max_pages_total:
            url, depth = queue.pop(0)
            # normalize drop fragments
            url = url.split('#')[0]
            if not is_allowed_url(url):
                print(f"[SKIP-NOT-ALLOWED] {url}")
                seen.add(url)
                continue
            if is_image_url(url):
                print(f"[SKIP-IMG] skipping image URL: {url}")
                seen.add(url)
                continue
            if url in seen:
                continue
            if depth > max_depth:
                continue

            print(f"[CRAWL] depth={depth} url={url}")
            try:
                try:
                    resp = await crawler.arun(url=url, max_pages=1)
                except TypeError:
                    resp = await crawler.arun(url=url)
            except Exception as e:
                print(f"  -> crawl error for {url}: {e}")
                seen.add(url)
                await asyncio.sleep(rate_delay)
                continue

            pages = getattr(resp, "results", None) or getattr(resp, "raw", None) or []
            if not pages:
                pages = [{"url": url, "markdown": getattr(resp, "markdown", "")}]

            for page in pages:
                pu = (page.get("url") or url).split('#')[0]
                if not is_allowed_url(pu):
                    print(f"[SKIP-NOT-ALLOWED-PAGE] {pu}")
                    seen.add(pu)
                    continue
                if is_image_url(pu):
                    print(f"[SKIP-IMG-PAGE] skipping image page: {pu}")
                    seen.add(pu)
                    continue
                if pu in seen:
                    continue
                seen.add(pu)
                title = page.get("title") or ""
                md = page.get("markdown") or page.get("text") or ""
                cleaned_md, contact_snippet = clean_markdown(md)
                cleaned_md = remove_image_urls(cleaned_md)
                results.append({"url": pu, "title": title, "markdown": cleaned_md, "contact": contact_snippet})

                if depth < max_depth:
                    links = extract_internal_links(md, allowed_prefix=ALLOWED_PREFIX)
                    for link in links:
                        if not is_allowed_url(link):
                            continue
                        if is_image_url(link):
                            continue
                        if link not in seen:
                            queue.append((link, depth + 1))

                await asyncio.sleep(rate_delay)

            await asyncio.sleep(rate_delay * 0.5)
    return results

def normalize_email(e: str) -> str:
    return e.strip().lower()
def normalize_phone(p: str) -> str:
    p = re.sub(r'[^\d\+]', '', p)
    return p
def normalize_url(u: str) -> str:
    u = u.strip()
    try:
        parsed = urlparse(u)
        scheme = parsed.scheme or "https"
        netloc = parsed.netloc.lower()
        path = parsed.path or ""
        qs = ("?" + parsed.query) if parsed.query else ""
        return f"{scheme}://{netloc}{path}{qs}"
    except Exception:
        return u.lower()

async def main():
    print("Starting BFS crawl from:", SEED_URL)
    pages = await bfs_crawl()

    print(f"Crawled {len(pages)} pages. Building dedupe and contact aggregates...")

    contact_info = {"emails": set(), "phones": set(), "contact_pages": set(), "addresses": set(), "texts": set()}
    dedup = {}
    for p in pages:
        dedup[p["url"]] = p
        c = p.get("contact", "") or ""
        if not c:
            continue
        for e in set(re.findall(_contact_email_re, c)):
            contact_info["emails"].add(normalize_email(e))
        for ph in set(re.findall(_contact_phone_re, c)):
            phn = normalize_phone(ph)
            if len(re.sub(r'\D', '', phn)) >= 7:
                contact_info["phones"].add(phn)
        contact_links = set(re.findall(r'\[.*?\]\((https?://[^\)]+contact[^\)]*)\)', c, flags=re.I))
        contact_rel_links = set(re.findall(r'\[.*?\]\((/[^)\s]*contact[^)]*)\)', c, flags=re.I))
        for rl in contact_rel_links:
            contact_links.add(urljoin(ALLOWED_PREFIX, rl))
        for cl in contact_links:
            if cl.startswith(ALLOWED_PREFIX):
                contact_info["contact_pages"].add(normalize_url(cl))
        for addr_candidate in re.findall(r'(.{0,120}\d{1,4}\s+[A-Za-z][A-Za-z0-9\.,\-\s]{3,120})', c):
            addr = addr_candidate.strip()
            if re.search(_contact_email_re, addr) or re.search(_contact_phone_re, addr):
                continue
            addr_norm = re.sub(r'\s{2,}', ' ', addr)
            if len(addr_norm) > 6:
                contact_info["addresses"].add(addr_norm)
        for line in c.splitlines():
            s = line.strip()
            if not s:
                continue
            if re.fullmatch(_contact_email_re, s) or re.fullmatch(_contact_phone_re, s):
                continue
            if re.search(r'https?://', s):
                continue
            if 3 <= len(s) <= 240:
                contact_info["texts"].add(s)

    final_pages = list(dedup.values())
    print(f"Final pages after dedupe: {len(final_pages)}")

    with open(RAW_OUT, "w", encoding="utf8") as fo:
        for p in final_pages:
            fo.write(json.dumps({"url": p["url"], "title": p.get("title", ""), "markdown": p.get("markdown", "")}, ensure_ascii=False) + "\n")
    print("Wrote cleaned pages to:", RAW_OUT)

    total_chunks = 0
    with open(CHUNKS_OUT, "w", encoding="utf8") as fc:
        for p in final_pages:
            text = p.get("markdown", "")
            chunks = chunk_text(text, max_chars=CHUNK_MAX_CHARS)
            for i, c in enumerate(chunks):
                chunk_doc = {"id": f"{p['url']}#chunk-{i}", "url": p["url"], "title": p.get("title", "") or "", "text": c}
                fc.write(json.dumps(chunk_doc, ensure_ascii=False) + "\n")
                total_chunks += 1

        has_any = any(contact_info[k] for k in contact_info)
        if has_any:
            parts = []
            if contact_info["emails"]:
                parts.append("Emails:\n" + "\n".join(sorted(contact_info["emails"])))
            if contact_info["phones"]:
                parts.append("Phone numbers:\n" + "\n".join(sorted(contact_info["phones"])))
            if contact_info["contact_pages"]:
                parts.append("Contact pages:\n" + "\n".join(sorted(contact_info["contact_pages"])))
            if contact_info["addresses"]:
                parts.append("Addresses:\n" + "\n".join(sorted(contact_info["addresses"])))
            if contact_info["texts"]:
                parts.append("Other contact snippets:\n" + "\n".join(sorted(contact_info["texts"])))
            contact_combined = "\n\n---\n\n".join(parts).strip()
            contact_chunks = chunk_text(contact_combined, max_chars=CHUNK_MAX_CHARS)
            if not contact_chunks:
                contact_chunks = [contact_combined] if contact_combined else []
            for idx, cc in enumerate(contact_chunks):
                contact_doc = {"id": f"itnb_contact_info#part-{idx}", "url": SEED_URL, "title": "Contact Information (aggregated)", "text": cc}
                fc.write(json.dumps(contact_doc, ensure_ascii=False) + "\n")
                total_chunks += 1
            print(f"Wrote aggregated deduplicated contact info as {len(contact_chunks)} chunk(s): id=itnb_contact_info#part-0..{len(contact_chunks)-1}")

    print(f"Wrote {total_chunks} chunks to:", CHUNKS_OUT)
    print("DONE.")

if __name__ == "__main__":
    asyncio.run(main())
