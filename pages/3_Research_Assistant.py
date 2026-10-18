"""VerityAI — Research Assistant page: ask questions about the knowledge base."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src import config
from src.rag.generation import answer_research_question
from src.rag.retrieval import HybridRetriever
from src.rag.vector_store import VectorStore

st.set_page_config(page_title=f"{config.APP_NAME} — Research Assistant", page_icon="📚", layout="wide")
st.title("📚 Research Assistant")
st.caption(
    "Ask a question about AI-generated media detection. Answers are retrieved "
    "passages with citations — never a free-form, uncited guess."
)


@st.cache_resource(show_spinner=False)
def _get_store_and_retriever():
    store = VectorStore()
    if store.count() == 0:
        return store, None
    return store, HybridRetriever(store)


store, retriever = _get_store_and_retriever()

if store.count() == 0:
    st.warning(
        "The knowledge base index is empty. Run `python scripts/ingest_documents.py` "
        "from the project root to index the documents in `knowledge_base/`, then "
        "reload this page."
    )
    st.stop()

st.caption(f"Knowledge base currently has {store.count()} indexed chunk(s).")

example_questions = [
    "Why do detectors trained on GAN images struggle with diffusion-model images?",
    "What's the difference between detection and content provenance (C2PA)?",
    "Can frame-level video analysis detect temporal deepfake artifacts?",
    "Why is CIFAKE not fully representative of real-world detector performance?",
]
chosen_example = st.selectbox("Try an example question, or type your own below:",
                               ["(type your own)"] + example_questions)

default_text = "" if chosen_example == "(type your own)" else chosen_example
question = st.text_input("Your question", value=default_text, placeholder="Ask about AI-media detection research...")

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Searching the knowledge base..."):
        evidence = retriever.retrieve(question, top_k=config.RETRIEVAL_TOP_K)
        answer = answer_research_question(question, evidence)

    st.divider()
    if answer.no_evidence_notice:
        st.warning(answer.no_evidence_notice)
    else:
        st.subheader("Evidence found")
        for i, (finding, citation) in enumerate(zip(answer.research_findings, answer.citations), 1):
            with st.expander(f"[{i}] {citation.format()}", expanded=(i <= 2)):
                st.write(finding)
                if citation.source_url:
                    st.markdown(f"[Open original source]({citation.source_url})")

        st.subheader("Citations")
        for c in answer.citations:
            st.markdown(f"- {c.format()}")
