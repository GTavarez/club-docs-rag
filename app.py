"""Day 25 - minimal Streamlit UI over the RAG app. Input box + answer. Nothing else."""
import csv
import datetime
import os
import pathlib

import streamlit as st

# Deployed: secrets live in Streamlit's secrets store, not in the code.
# Must be copied into the environment BEFORE importing rag (it reads env at import).
try:
    for k in ("GEMINI_API_KEY", "RAG_PROVIDER", "CHAT_MODEL", "EMBED_MODEL", "ADMIN_KEY"):
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

FEEDBACK = pathlib.Path(__file__).with_name("feedback_log.csv")


def log_row(q, out, vote=""):
    """Append one row. Also print() so it shows in Streamlit Cloud's app logs
    (the cloud filesystem is wiped on restart; the logs are the backup)."""
    new = not FEEDBACK.exists()
    row = [datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           q, out["found"], out["answer"] if out["found"] else "", vote]
    try:
        with FEEDBACK.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time_utc", "question", "found", "answer", "vote"])
            w.writerow(row)
    except OSError:
        pass
    print("FEEDBACK_ROW", row, flush=True)


# Owner-only view: open the app URL with ?admin=<ADMIN_KEY> to see/download the log.
if os.environ.get("ADMIN_KEY") and st.query_params.get("admin") == os.environ["ADMIN_KEY"]:
    st.subheader("Feedback log (owner view)")
    if FEEDBACK.exists():
        st.dataframe(csv.DictReader(FEEDBACK.open(encoding="utf-8")))
        st.download_button("Download CSV", FEEDBACK.read_bytes(), "feedback_log.csv")
    else:
        st.write("No rows yet.")

st.caption("Questions and thumbs up/down are logged to improve the app. Don't type private info.")
q = st.text_input("Ask a question about the club")
if q.strip():
    if st.session_state.get("last_q") != q:  # buttons rerun the script; don't re-call the model
        with st.spinner("Thinking..."):
            try:
                out = ask_json(q)
            except Exception as e:
                st.error(f"Service problem (not a wrong answer): {type(e).__name__}")
                st.stop()
        st.session_state.update(last_q=q, last_out=out, voted=False)
        log_row(q, out)
    out = st.session_state["last_out"]
    if out["found"]:
        st.write(out["answer"])
        st.caption("Sources: " + ", ".join(out["sources"]))
    else:
        st.info("I don't know - that isn't in the club documents.")
    if not st.session_state.get("voted"):
        st.write("Was this helpful?")
        c1, c2, _ = st.columns([1, 1, 6])
        vote = "up" if c1.button("👍") else "down" if c2.button("👎") else ""
        if vote:
            log_row(q, out, vote)
            st.session_state["voted"] = True
            st.rerun()
    else:
        st.success("Thanks - logged.")
