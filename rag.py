"""Week 3 — minimal RAG loop over data/docs/*.txt using local Ollama.

Documents -> chunks -> embeddings (nomic-embed-text) -> chromadb search
-> stuffed into a prompt -> answer (llama3.2). No framework on purpose.
"""
import glob
import os
import random
import re
import time

import chromadb
from openai import APIConnectionError, InternalServerError, OpenAI, RateLimitError

# --- provider switch -------------------------------------------------------
# RAG_PROVIDER=ollama (default, local)  |  RAG_PROVIDER=gemini (hosted, deployable)
# Everything downstream is unchanged: both speak the OpenAI API.
PROVIDER = os.getenv("RAG_PROVIDER", "ollama").lower()

if PROVIDER == "gemini":
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("RAG_PROVIDER=gemini but GEMINI_API_KEY is not set")
    client = OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=key,
    )
    CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.6-flash")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
else:
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

db = chromadb.Client()
# separate collection per provider: embedding dimensions differ between them
col = db.get_or_create_collection(f"docs_{PROVIDER}")


RETRYABLE = (RateLimitError, InternalServerError, APIConnectionError)


def _suggested_wait(err):
    """Google tells us how long to wait; use it instead of guessing."""
    m = re.search(r"retryDelay['\"]?:\s*['\"](\d+)s", str(err)) or \
        re.search(r"retry in (\d+)", str(err))
    return int(m.group(1)) + 1 if m else None


def call_with_backoff(fn, *, tries=6, **kwargs):
    """Retry 429/503 with exponential backoff. Raises if it never succeeds,
    so a quota problem can never be silently scored as a wrong answer."""
    delay = 2
    for attempt in range(tries):
        try:
            return fn(**kwargs)
        except RETRYABLE as e:
            if attempt == tries - 1:
                raise
            wait = _suggested_wait(e) or delay
            if wait > 120:              # daily quota, not a momentary spike
                raise RuntimeError(
                    f"Provider asked us to wait {wait}s (~{wait / 3600:.1f}h): "
                    "daily quota is exhausted, not a transient rate limit."
                ) from e
            print(f"    [retry {attempt + 1}/{tries - 1}] {type(e).__name__}; waiting {wait}s")
            time.sleep(wait + random.uniform(0, 1))
            delay = min(delay * 2, 60)


def chat(**kwargs):
    """Chat completion that survives free-tier rate limits."""
    return call_with_backoff(client.chat.completions.create, **kwargs)


def embed(texts):
    r = call_with_backoff(client.embeddings.create, model=EMBED_MODEL, input=texts)
    return [d.embedding for d in r.data]


def chunk(text, size=800, overlap=100):
    return [text[i:i + size] for i in range(0, len(text), size - overlap)]


def index(pattern=None):
    pattern = pattern or os.getenv("DOCS_GLOB", "data/docs/*.txt")
    ids, docs, metas = [], [], []
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for i, c in enumerate(chunk(text)):
            ids.append(f"{name}#{i}")
            docs.append(c)
            metas.append({"source": name})
    col.add(ids=ids, documents=docs, embeddings=embed(docs), metadatas=metas)
    print(f"indexed {len(docs)} chunks from {len(set(m['source'] for m in metas))} files")


SYSTEM = (
    "Answer only from the provided context. "
    "If the answer is not in the context, say exactly: I don't know. "
    "End your answer with the source name(s) you used, in brackets."
)


def retrieve(q, k=3):
    hits = col.query(query_embeddings=embed([q]), n_results=k)
    return list(zip(hits["metadatas"][0], hits["documents"][0]))


def ask(q, k=3, show_context=False):
    hits = retrieve(q, k)
    ctx = "\n\n".join(f"[{m['source']}] {d}" for m, d in hits)
    if show_context:
        print("--- retrieved ---")
        for m, _ in hits:
            print("  ", m["source"])
        print("-----------------")
    r = chat(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {q}"},
        ],
    )
    return r.choices[0].message.content


if __name__ == "__main__":
    index()
    print("Ask a question (blank or 'exit' to quit). Prefix with '?' to see what was retrieved.")
    while True:
        q = input("\nQ> ").strip()
        if q in ("", "exit"):
            break
        show = q.startswith("?")
        print(ask(q.lstrip("? "), show_context=show))
