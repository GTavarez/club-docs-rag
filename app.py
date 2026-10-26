"""Day 25 - minimal Streamlit UI over the RAG app. Input box + answer. Nothing else."""
import os

import streamlit as st

# Deployed: secrets live in Streamlit's secrets store, not in the code.
# Must be copied into the environment BEFORE importing rag (it reads env at import).
try:
    for k in ("GEMINI_API_KEY", "RAG_PROVIDER", "CHAT_MODEL", "EMBED_MODEL"):
        if k in st.secrets:
            os.environ[k] = str(st.secrets[k])
except Exception:
    pass  # no secrets file locally; use real environment variables
os.environ.setdefault("RAG_PROVIDER", "gemini")

from drill_json import ask_json  # noqa: E402
from rag import CHAT_MODEL, PROVIDER, index  # noqa: E402

st.set_page_config(page_title="Club Docs Q&A", page_icon="📄")
st.title("Club Docs Q&A")
st.caption(f"Answers only from the club's documents. Model: {PROVIDER}/{CHAT_MODEL}")


@st.cache_resource(show_spinner="Indexing documents (first load only)...")
def build_index():
    index()
    return True


build_index()

q = st.text_input("Ask a question about the club")
if q.strip():
    with st.spinner("Thinking..."):
        try:
            out = ask_json(q)
        except Exception as e:
            st.error(f"Service problem (not a wrong answer): {type(e).__name__}")
            st.stop()
    if out["found"]:
        st.write(out["answer"])
        st.caption("Sources: " + ", ".join(out["sources"]))
    else:
        st.info("I don't know - that isn't in the club documents.")
