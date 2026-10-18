"""VerityAI — Image Analysis page.

The core end-to-end flow: upload -> validate -> real detector inference ->
hybrid RAG retrieval -> evidence-grounded report -> download.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src import config, validation, reporting
from src.detector.base import DetectorError
from src.detector.image_detector import predict_image
from src.rag.generation import generate_explanation
from src.rag.retrieval import HybridRetriever
from src.rag.bootstrap import ensure_index_built
from src.rag.vector_store import VectorStore

st.set_page_config(page_title=f"{config.APP_NAME} — Image Analysis", page_icon="🖼️", layout="wide")
st.title("🖼️ Image Analysis")
st.caption(
    "Real pretrained-model inference, every time. No prediction shown here is "
    "random, simulated, or hardcoded."
)


@st.cache_resource(show_spinner=False)
def _get_retriever() -> HybridRetriever | None:
    try:
        store = VectorStore()
        ensure_index_built(store)
        if store.count() == 0:
            return None
        return HybridRetriever(store)
    except Exception:  # noqa: BLE001 - RAG is best-effort; detector result still shown
        return None


uploaded = st.file_uploader(
    "Drag and drop an image, or click to browse",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=False,
    help=f"Max {config.MAX_IMAGE_SIZE_MB}MB. Supported: JPG, JPEG, PNG, WEBP.",
)

if uploaded is None:
    st.info("Upload an image above to begin.")
    st.stop()

file_bytes = uploaded.getvalue()
validation_result = validation.validate_image_bytes(file_bytes, uploaded.name)

if not validation_result.ok:
    # Validate BEFORE any rendering: st.image would raise on a non-image file.
    st.error(f"❌ Validation failed: {validation_result.reason}")
    st.stop()

col_preview, col_status = st.columns([1, 1])
with col_preview:
    st.image(file_bytes, caption=uploaded.name, use_column_width=True)

with col_status:
    st.success("✅ File validated")
    st.markdown(
        f"- **Detected type:** {validation_result.detected_mime}\n"
        f"- **Dimensions:** {validation_result.width}×{validation_result.height}px\n"
        f"- **Size:** {len(file_bytes) / 1024:.1f} KB"
    )

temp_path = validation.safe_temp_path(Path(uploaded.name).suffix.lower())
try:
    temp_path.write_bytes(file_bytes)

    with st.spinner("Running detector inference..."):
        try:
            detection = predict_image(temp_path)
        except DetectorError as exc:
            st.error(f"❌ Detector error: {exc}")
            st.caption(
                "This is a real failure, not a simulated result — the analysis did "
                "not run. Check that dependencies are installed and (for the first "
                "run) that this machine has internet access to download the model."
            )
            st.stop()

    category_color = {
        "AI-generated suspected": "🔴",
        "Real or non-synthetic suspected": "🟢",
        "Inconclusive": "🟡",
    }
    st.divider()
    st.subheader(f"{category_color.get(detection.category.value, '⚪')} {detection.category.value}")

    score_cols = st.columns(len(detection.scores) or 1)
    for col, (label, score) in zip(score_cols, detection.scores.items()):
        col.metric(label.upper(), f"{score:.1%}")

    with st.expander("Detector details", expanded=False):
        st.markdown(
            f"- **Model:** {detection.model_version_label}\n"
            f"- **Raw top label:** {detection.raw_top_label}\n"
            f"- **Inference latency:** {detection.inference_latency_ms:.0f} ms\n"
            f"- **Decision policy:** {detection.decision_policy}"
        )

    st.divider()
    st.subheader("📚 Research-backed explanation")
    retriever = _get_retriever()
    query = f"{detection.category.value} {detection.raw_top_label} AI image detector limitations generalization"
    evidence = retriever.retrieve(query, top_k=config.RETRIEVAL_TOP_K) if retriever else []
    explanation = generate_explanation(detection, evidence, query_context=detection.category.value)

    st.write(explanation.detector_summary)
    if explanation.research_findings:
        for finding in explanation.research_findings:
            st.markdown(f"- {finding}")
    elif explanation.no_evidence_notice:
        st.warning(explanation.no_evidence_notice)

    with st.expander("⚠️ Limitations and uncertainty", expanded=True):
        for lim in explanation.limitations:
            st.markdown(f"- {lim}")

    st.divider()
    report = reporting.build_image_report(
        file_name=uploaded.name,
        file_type=validation_result.detected_mime,
        file_size_bytes=len(file_bytes),
        image_width=validation_result.width,
        image_height=validation_result.height,
        detection=detection,
        explanation=explanation,
    )

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "⬇️ Download report (JSON)",
            data=reporting.to_json(report),
            file_name=f"verityai_report_{int(time.time())}.json",
            mime="application/json",
            use_container_width=True,
        )
    with dl_col2:
        st.download_button(
            "⬇️ Download report (Markdown)",
            data=reporting.to_markdown(report),
            file_name=f"verityai_report_{int(time.time())}.md",
            mime="text/markdown",
            use_container_width=True,
        )

finally:
    validation.cleanup_temp_file(temp_path)
