"""VerityAI — Home / Overview page."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from src import config

st.set_page_config(
    page_title=f"{config.APP_NAME} — Home",
    page_icon="🔎",
    layout="wide",
)

CUSTOM_CSS = """
<style>
.verity-hero {
    padding: 1.75rem 2rem;
    border-radius: 12px;
    background: linear-gradient(135deg, #F0FBFA 0%, #F3F6F6 100%);
    border: 1px solid #DCEDEB;
    margin-bottom: 1.25rem;
}
.verity-tagline { color: #0F766E; font-weight: 600; font-size: 1.05rem; }
.verity-card {
    border: 1px solid #E3E8E8;
    border-radius: 10px;
    padding: 1.1rem 1.25rem;
    background: #FFFFFF;
    height: 100%;
}
.verity-card h4 { margin-top: 0; color: #0B2B33; }
.verity-limit {
    border-left: 4px solid #D97706;
    background: #FFFBEB;
    padding: 0.9rem 1.1rem;
    border-radius: 6px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="verity-hero">
        <h1>🔎 {config.APP_NAME}</h1>
        <p class="verity-tagline">{config.APP_TAGLINE}</p>
        <p>An evidence-grounded screening tool for AI-generated images and video —
        combining a real pretrained detector with Retrieval-Augmented Generation
        over AI-media-forensics research, so every result comes with its own
        supporting evidence and disclosed limitations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        """<div class="verity-card"><h4>🖼️ Image Analysis</h4>
        <p>Upload a JPG/PNG/WEBP image and get a real model prediction, raw
        scores, a research-backed explanation, and a downloadable report.</p>
        </div>""",
        unsafe_allow_html=True,
    )
    st.page_link("pages/1_Image_Analysis.py", label="Analyze an image →")
with col2:
    st.markdown(
        """<div class="verity-card"><h4>🎞️ Video Analysis</h4>
        <p>Experimental: samples evenly-spaced frames from a short MP4/MOV
        and runs the same image detector on each, with a clearly disclosed
        aggregation method.</p>
        </div>""",
        unsafe_allow_html=True,
    )
    st.page_link("pages/2_Video_Analysis.py", label="Analyze a video →")
with col3:
    st.markdown(
        """<div class="verity-card"><h4>📚 Research Assistant</h4>
        <p>Ask questions about AI-media detection and get answers grounded
        in retrieved passages, with page-level citations — never a
        free-form guess.</p>
        </div>""",
        unsafe_allow_html=True,
    )
    st.page_link("pages/3_Research_Assistant.py", label="Ask a question →")

st.divider()

st.subheader("How the detection process works")
st.markdown(
    """
    1. **You upload** an image or short video. It's validated (type, size,
       dimensions/duration) and never written anywhere outside a temporary
       folder that's cleaned up after analysis.
    2. **A real pretrained model** classifies the image (or each sampled
       video frame) — see the About page for exactly which model, and why.
       There is no random or hardcoded output anywhere in this pipeline.
    3. **The knowledge base is searched** (hybrid keyword + semantic search
       over AI-media-forensics research) for passages relevant to the
       result and the detector's known limitations.
    4. **A report is assembled** that keeps the model's output and the
       retrieved research strictly separate and cited, states the exact
       decision policy used, and lists limitations and uncertainty —
       downloadable as JSON or Markdown.
    """
)

st.markdown(
    """
    <div class="verity-limit">
    <strong>Limitations, upfront:</strong> VerityAI is an educational screening
    tool, not a forensic or legal authority. It cannot prove a file's authenticity
    or provenance. Its detector was trained on a specific, disclosed dataset and
    will be less reliable on generators released after that training data was
    collected. See the <strong>About / Methodology</strong> page for full details
    before relying on any result.
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Uploaded media is processed only in this app's local/session temporary "
    "storage for the duration of your analysis, then deleted — see About → "
    "Privacy for details."
)
