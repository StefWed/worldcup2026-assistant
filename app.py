"""
Streamlit front end for the World Cup team-analysis agent.

Run with:
    streamlit run app.py
"""

import os
import tempfile

import streamlit as st

# Make secrets available as env vars before config.py builds the OpenAI client.
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

from rag import run_pipeline
from tools import init_document_search
from agent import run_agent

st.set_page_config(page_title="World Cup Scout Agent", page_icon="⚽", layout="wide")

st.title("⚽ World Cup Team Scout Agent")
st.caption(
    "Ask about team playing styles, current tournament news, or get a simple "
    "match prediction. The agent decides which tool to use."
)

# --- Session state ---
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"question": ..., "answer": ..., "trace": ...}

# --- Sidebar: build / rebuild the document index ---
with st.sidebar:
    st.header("1. Load team descriptions")
    uploaded_pdf = st.file_uploader("Upload the team-description PDF", type=["pdf"])

    if uploaded_pdf is not None and st.button("Build index", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_pdf.read())
            tmp_path = tmp_file.name

        with st.spinner("Loading pages, embedding, and building the vector index..."):
            index, documents, page_warnings = run_pipeline(tmp_path)
            init_document_search(index, documents)

        st.session_state.index_ready = True
        st.session_state.teams = [doc.metadata["team"] for doc in documents]

        if page_warnings:
            for w in page_warnings:
                st.warning(w)
        st.success(f"Index built for {len(documents)} team page(s).")

    if st.session_state.index_ready:
        st.markdown("**Teams loaded:**")
        st.write(", ".join(st.session_state.get("teams", [])))
    else:
        st.info("Upload a PDF and click 'Build index' to enable document search.")

    st.divider()
    st.header("2. About the tools")
    st.markdown(
        "- **search_document** — team identity/style, from the PDF\n"
        "- **search_web** — current news, results, standings\n"
        "- **analyze_match** — quick stats-based prediction\n"
    )

# --- Main: chat interface ---
st.header("Ask the agent")

example_cols = st.columns(3)
example_questions = [
    "What makes Spain dangerous at this World Cup?",
    "How do Switzerland and Norway differ tactically?",
    "Predict France vs Argentina.",
]
clicked_example = None
for col, q in zip(example_cols, example_questions):
    if col.button(q):
        clicked_example = q

question = st.text_input("Your question", value=clicked_example or "")

if st.button("Ask") and question.strip():
    if not st.session_state.index_ready:
        st.warning(
            "No document index yet — search_document won't have anything to "
            "search. You can still ask web-search or prediction questions."
        )

    with st.spinner("Thinking..."):
        result = run_agent(question)

    st.session_state.history.insert(0, {
        "question": question,
        "answer": result["answer"],
        "trace": result["trace"],
    })

# --- History ---
for item in st.session_state.history:
    st.markdown(f"### Q: {item['question']}")
    st.markdown(item["answer"])

    if item["trace"]:
        with st.expander("Show tool activity"):
            for step in item["trace"]:
                st.markdown(f"**Tool called:** `{step['tool']}`")
                st.json(step["args"])
                st.text(step["result"][:1000])
                st.divider()
    st.divider()
