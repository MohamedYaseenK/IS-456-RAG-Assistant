"""
app.py
------
Streamlit frontend for the IS 456 RAG chatbot.

Run with:
    streamlit run app.py
"""

import time
import streamlit as st

from src.pipeline import RAGPipeline


# ---------- Config ----------
MAX_QUERIES_PER_SESSION = 10
MIN_SECONDS_BETWEEN_QUERIES = 3   # simple client-side throttle to protect API quota

st.set_page_config(
    page_title="IS 456 RAG Assistant",
    page_icon="🏗️",
    layout="centered",
)


# ---------- Cached pipeline (loaded once per server process, not per session) ----------
@st.cache_resource(show_spinner="Loading knowledge base...")
def get_pipeline():
    pipeline = RAGPipeline()
    pipeline.build()
    return pipeline


# ---------- Session state ----------
if "history" not in st.session_state:
    st.session_state.history = []
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "last_query_time" not in st.session_state:
    st.session_state.last_query_time = 0.0


# ---------- Header ----------
st.title("🏗️ IS 456 RAG Assistant")
st.caption(
    "Ask questions about IS 456:2000 — the Indian Standard code for Plain and "
    "Reinforced Concrete. Answers are retrieved from the source document and "
    "cite page numbers."
)

with st.expander("ℹ️ About this demo / usage limits"):
    st.markdown(f"""
    - This is a **portfolio demo**, not an official engineering reference — always verify
      critical decisions against the actual IS 456:2000 standard.
    - Limited to **{MAX_QUERIES_PER_SESSION} questions per session** to protect free-tier API quota.
    - Please wait a few seconds between questions.
    - Source: IS 456:2000, publicly available via the Bureau of Indian Standards
      right-to-information disclosure.
    """)

remaining = MAX_QUERIES_PER_SESSION - st.session_state.query_count
st.progress(
    st.session_state.query_count / MAX_QUERIES_PER_SESSION,
    text=f"{remaining} question(s) remaining this session"
)

st.divider()


# ---------- Chat history ----------
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["query"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        with st.expander("📄 Sources"):
            for src in turn["sources"]:
                st.markdown(f"**Page {src['page_no']}** (distance: {src['distance']:.3f})")
                st.caption(src["text"][:300] + "...")


# ---------- Input ----------
query = st.chat_input("Ask a question about IS 456...")

if query:
    # --- usage limit check ---
    if st.session_state.query_count >= MAX_QUERIES_PER_SESSION:
        st.error(
            f"⚠️ You've reached the limit of {MAX_QUERIES_PER_SESSION} questions "
            "for this session. Please refresh the page to start a new session."
        )
        st.stop()

    # --- throttle check ---
    elapsed = time.time() - st.session_state.last_query_time
    if elapsed < MIN_SECONDS_BETWEEN_QUERIES:
        st.warning(
            f"⏳ Please wait {MIN_SECONDS_BETWEEN_QUERIES - elapsed:.1f} more "
            "second(s) before asking another question."
        )
        st.stop()

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching IS 456 and generating answer..."):
            try:
                pipeline = get_pipeline()
                result = pipeline.ask(query)
                answer = result["answer"]
                sources = result["sources"]
            except Exception as e:
                answer = (
                    "⚠️ Sorry, something went wrong generating a response "
                    "(likely an API quota limit on this free demo). "
                    "Please try again in a minute."
                )
                sources = []
                st.exception(e)

        st.markdown(answer)
        if sources:
            with st.expander("📄 Sources"):
                for src in sources:
                    st.markdown(f"**Page {src['page_no']}** (distance: {src['distance']:.3f})")
                    st.caption(src["text"][:300] + "...")

    st.session_state.history.append({"query": query, "answer": answer, "sources": sources})
    st.session_state.query_count += 1
    st.session_state.last_query_time = time.time()
    st.rerun()


# ---------- Footer ----------
st.divider()
st.caption("Built as a portfolio RAG project · Source: IS 456:2000 · Not for professional engineering use")