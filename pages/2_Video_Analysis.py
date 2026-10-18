"""VerityAI — Video Analysis page (experimental).

Frame-level analysis only. See knowledge_base/03_video_and_deepfake_detection.md
for why this is explicitly NOT a temporal/video-level deepfake detector.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src import config, validation, reporting
from src.detector.base import DetectorError
from src.detector.video_analyzer import analyze_video, validate_video_file
from src.rag.generation import generate_explanation
from src.rag.retrieval import HybridRetriever
from src.rag.vector_store import VectorStore

st.set_page_config(page_title=f"{config.APP_NAME} — Video Analysis", page_icon="🎞️", layout="wide")
st.title("🎞️ Video Analysis")
st.warning(
    "**Experimental — frame-level only.** This samples a handful of individual "
    "frames and classifies each with the same image detector. It is **not** a "
    "temporal/video-level deepfake detector, and inconsistent frame scores are "
    "never treated as proof of manipulation. See About → Methodology.",
    icon="⚠️",
)


@st.cache_resource(show_spinner=False)
def _get_retriever() -> HybridRetriever | None:
    try:
        store = VectorStore()
        if store.count() == 0:
            return None
        return HybridRetriever(store)
    except Exception:  # noqa: BLE001
        return None


uploaded = st.file_uploader(
    "Upload a short video",
    type=["mp4", "mov"],
    accept_multiple_files=False,
    help=f"Max {config.MAX_VIDEO_SIZE_MB}MB, max {config.MAX_VIDEO_DURATION_SECONDS}s. "
         f"Up to {config.MAX_VIDEO_FRAMES_SAMPLED} frames will be sampled.",
)

if uploaded is None:
    st.info("Upload a video above to begin.")
    st.stop()

file_bytes = uploaded.getvalue()
validation_result = validation.validate_video_bytes(file_bytes, uploaded.name)
if not validation_result.ok:
    st.error(f"❌ Validation failed: {validation_result.reason}")
    st.stop()

temp_path = validation.safe_temp_path(Path(uploaded.name).suffix.lower())
frames_dir = config.TMP_UPLOAD_DIR / f"frames_{temp_path.stem}"

try:
    temp_path.write_bytes(file_bytes)

    video_validation = validate_video_file(temp_path)
    if not video_validation.ok:
        st.error(f"❌ {video_validation.reason}")
        st.caption(
            f"If this is just over the limit, trim the clip or reduce "
            f"MAX_VIDEO_DURATION_SECONDS in src/config.py (currently "
            f"{config.MAX_VIDEO_DURATION_SECONDS}s) — this is a conservative demo "
            f"default, not a hard technical ceiling."
        )
        st.stop()

    st.success(
        f"✅ Validated: {video_validation.duration_seconds:.1f}s, "
        f"{video_validation.frame_count} total frames, {video_validation.fps:.1f} fps"
    )

    with st.spinner(f"Sampling up to {config.MAX_VIDEO_FRAMES_SAMPLED} evenly-spaced frames and "
                     f"running the detector on each..."):
        try:
            result = analyze_video(temp_path, workdir=frames_dir)
        except DetectorError as exc:
            st.error(f"❌ Analysis failed: {exc}")
            st.caption("This is a real failure — not a simulated result.")
            st.stop()

    st.divider()
    category_color = {
        "AI-generated suspected": "🔴",
        "Real or non-synthetic suspected": "🟢",
        "Inconclusive": "🟡",
    }
    st.subheader(f"{category_color.get(result.category.value, '⚪')} {result.category.value} "
                 f"(experimental aggregate)")

    m1, m2, m3 = st.columns(3)
    m1.metric("Frames analyzed", result.frames_analyzed)
    m2.metric("Mean AI score", f"{result.mean_ai_score:.1%}")
    m3.metric("Mean Real score", f"{result.mean_real_score:.1%}")
    st.caption(f"Aggregation method: {result.aggregation_method}")

    st.subheader("Per-frame results")
    frame_cols = st.columns(min(4, len(result.frame_results)))
    for i, fr in enumerate(result.frame_results):
        with frame_cols[i % len(frame_cols)]:
            st.image(str(fr.frame_path), caption=f"t={fr.timestamp_seconds:.1f}s")
            top_label = fr.detection.raw_top_label
            top_score = max(fr.detection.scores.values())
            st.caption(f"{top_label}: {top_score:.1%}")

    st.divider()
    st.subheader("📚 Research-backed explanation")
    retriever = _get_retriever()
    query = (
        f"{result.category.value} video frame-level analysis temporal deepfake "
        f"detection limitations inconsistent frame scores"
    )
    evidence = retriever.retrieve(query, top_k=config.RETRIEVAL_TOP_K) if retriever else []
    from src.detector.base import DetectionResult, ResultCategory
    synthetic_summary_detection = DetectionResult(
        scores={"ai": result.mean_ai_score, "real": result.mean_real_score},
        category=result.category,
        model_name=result.frame_results[0].detection.model_name,
        model_version_label=result.frame_results[0].detection.model_version_label,
        decision_policy=f"Mean-score aggregation across {result.frames_analyzed} frames; "
                         f"{result.aggregation_method}",
        raw_top_label=("ai" if result.mean_ai_score >= result.mean_real_score else "real"),
        inference_latency_ms=sum(fr.detection.inference_latency_ms for fr in result.frame_results),
    )
    explanation = generate_explanation(synthetic_summary_detection, evidence, query_context=result.category.value)

    st.write(explanation.detector_summary)
    if explanation.research_findings:
        for finding in explanation.research_findings:
            st.markdown(f"- {finding}")
    elif explanation.no_evidence_notice:
        st.warning(explanation.no_evidence_notice)

    with st.expander("⚠️ Limitations and uncertainty", expanded=True):
        for lim in explanation.limitations:
            st.markdown(f"- {lim}")
        st.markdown(
            "- **Video-specific:** this is frame-level analysis, not a validated "
            "temporal/video-level detector. Frame-score variance alone is not "
            "evidence of manipulation (see About → Methodology)."
        )

    st.divider()
    report = reporting.build_video_report(
        file_name=uploaded.name,
        file_type=validation_result.detected_mime,
        file_size_bytes=len(file_bytes),
        video_duration_seconds=result.video_duration_seconds,
        frames_analyzed=result.frames_analyzed,
        frame_level_scores=[
            {"frame_index": fr.frame_index, "timestamp_seconds": fr.timestamp_seconds,
             "scores": fr.detection.scores, "top_label": fr.detection.raw_top_label}
            for fr in result.frame_results
        ],
        aggregation_method=result.aggregation_method,
        mean_ai_score=result.mean_ai_score,
        mean_real_score=result.mean_real_score,
        predicted_category=result.category.value,
        detector_name=result.frame_results[0].detection.model_name,
        detector_version=result.frame_results[0].detection.model_version_label,
        explanation=explanation,
    )

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "⬇️ Download report (JSON)", data=reporting.to_json(report),
            file_name=f"verityai_video_report_{int(time.time())}.json",
            mime="application/json", use_container_width=True,
        )
    with dl_col2:
        st.download_button(
            "⬇️ Download report (Markdown)", data=reporting.to_markdown(report),
            file_name=f"verityai_video_report_{int(time.time())}.md",
            mime="text/markdown", use_container_width=True,
        )

finally:
    validation.cleanup_temp_file(temp_path)
    if frames_dir.exists():
        import shutil
        shutil.rmtree(frames_dir, ignore_errors=True)
