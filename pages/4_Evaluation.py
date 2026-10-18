"""VerityAI — Evaluation Dashboard.

Shows only real, computed metrics. Anything without real evaluation data
behind it is explicitly rendered as "Not evaluated" — never a placeholder
number.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src import config

st.set_page_config(page_title=f"{config.APP_NAME} — Evaluation", page_icon="📊", layout="wide")
st.title("📊 Evaluation Dashboard")
st.caption(
    "Runs scripts/evaluate.py and shows its real output. If a metric reads "
    "'Not evaluated', the labeled data it needs isn't present — see README "
    "'Evaluation instructions'. Nothing on this page is a placeholder number."
)

eval_images_present = (config.PROJECT_ROOT / "data" / "eval" / "images" / "real").is_dir() and \
                       (config.PROJECT_ROOT / "data" / "eval" / "images" / "ai").is_dir()
rag_eval_present = (config.PROJECT_ROOT / "data" / "eval" / "rag_eval.jsonl").exists()

col1, col2 = st.columns(2)
col1.metric("Detector eval data found", "Yes" if eval_images_present else "No")
col2.metric("RAG eval data found", "Yes" if rag_eval_present else "No")

if not eval_images_present and not rag_eval_present:
    st.info(
        "No labeled evaluation data is present in this environment, so there is "
        "nothing real to show yet. See `data/README.md` and the main README's "
        "'Evaluation instructions' section for how to add a free, licensed "
        "evaluation set, then come back to this page."
    )

if st.button("Run evaluation now", type="primary"):
    with st.spinner("Running scripts/evaluate.py — this loads the detector model and "
                     "may take a while on first run..."):
        try:
            result = subprocess.run(
                [sys.executable, str(config.PROJECT_ROOT / "scripts" / "evaluate.py")],
                capture_output=True, text=True, timeout=600, cwd=str(config.PROJECT_ROOT),
            )
            output = result.stdout + ("\n" + result.stderr if result.returncode != 0 else "")
        except subprocess.TimeoutExpired:
            output = "Evaluation timed out after 10 minutes."
    st.code(output, language="text")

st.divider()
st.subheader("What each metric means")
st.markdown(
    """
    - **Precision / Recall / F1 / Confusion matrix / ROC-AUC** — detector
      classification quality, computed only against real labeled images
      (see `data/eval/images/{real,ai}/`).
    - **Recall@K / Mean Reciprocal Rank / Citation coverage** — RAG
      retrieval quality, computed only against a real labeled question set
      (see `data/eval/rag_eval.jsonl`).
    - **Processing latency** — measured wall-clock time per detector call,
      reported alongside the metrics above.
    - **Failure-case analysis** — individual examples the detector or
      retriever got wrong are printed by `scripts/evaluate.py` so they can
      be inspected, not just averaged away.

    Full definitions and the reasoning behind each metric are in
    `knowledge_base/06_evaluation_methodology.md` (also searchable from the
    Research Assistant page).
    """
)
