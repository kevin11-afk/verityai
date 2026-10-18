"""Image/video analysis: validation -> real detector -> hybrid retrieval ->
grounded explanation -> typed Report. Wraps existing src/ functions only."""
from __future__ import annotations

import hashlib
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from api.errors import ApiError
from api.schemas import (
    DetectorInfo, Evidence, Explanation, FileInfo, FrameInfo, ModelResult,
    Passage, Report, RetrievalDiag, Thresholds, VideoInfo,
)
from api.services.knowledge import KnowledgeService
from api.settings import APP_VERSION
from src import config, validation
from src.detector.base import DetectionResult, DetectorError, ResultCategory
from src.detector.image_detector import predict_image
from src.detector.video_analyzer import analyze_video, validate_video_file
from src.rag.generation import generate_explanation

UNCERTAINTY_WARNING = (
    "This result is an automated screening signal only. It does not prove whether the file is "
    "authentic, AI-generated, or edited, and must not be treated as a forensic or legal "
    "determination. Always corroborate with other evidence."
)

# At most two inferences at once: keeps peak memory predictable on small hosts.
_inference_slots = threading.BoundedSemaphore(2)


def _thresholds() -> Thresholds:
    return Thresholds(ai=config.AI_SUSPECTED_THRESHOLD, real=config.REAL_SUSPECTED_THRESHOLD)


def _evidence(kb: KnowledgeService, query: str) -> Evidence:
    passages, diag = kb.retrieve(query, top_k=config.RETRIEVAL_TOP_K)
    return Evidence(
        passages=[
            Passage(
                citation_id=i, chunk_id=p.chunk_id, title=p.title, source_file=p.source_file,
                page_number=p.page_number, source_url=p.source_url, text=p.text,
                vector_rank=p.vector_rank, bm25_rank=p.bm25_rank, rrf_score=p.rrf_score,
                vector_distance=p.vector_distance,
            )
            for i, p in enumerate(passages, start=1)
        ],
        diagnostics=RetrievalDiag(**diag.__dict__),
    ), passages


def analyze_image_bytes(data: bytes, filename: str, kb: KnowledgeService) -> Report:
    check = validation.validate_image_bytes(data, filename)
    if not check.ok:
        raise ApiError(422, "invalid_file", check.reason or "The image failed validation.")

    temp = validation.safe_temp_path(Path(filename).suffix.lower())
    try:
        temp.write_bytes(data)
        with _inference_slots:
            try:
                detection = predict_image(temp)
            except DetectorError as exc:
                raise ApiError(503, "detector_unavailable", str(exc)) from exc
    finally:
        validation.cleanup_temp_file(temp)

    query = f"{detection.category.value} {detection.raw_top_label} AI image detector limitations generalization"
    evidence, passages = _evidence(kb, query)
    explanation = generate_explanation(detection, passages, query_context=detection.category.value)

    return Report(
        id=uuid.uuid4().hex,
        created_at=datetime.now(timezone.utc).isoformat(),
        kind="image",
        app_version=APP_VERSION,
        file=FileInfo(
            name=Path(filename).name[:200], mime=check.detected_mime or "image/*", size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(), width=check.width, height=check.height,
        ),
        detector=DetectorInfo(name=detection.model_name, version_label=detection.model_version_label,
                              thresholds=_thresholds()),
        result=ModelResult(
            category=detection.category.value, predicted_class=detection.raw_top_label,
            scores=detection.scores, decision_policy=detection.decision_policy,
            latency_ms=detection.inference_latency_ms,
        ),
        analysis_method="Single-pass classification with a pretrained Swin-Tiny image classifier",
        evidence=evidence,
        explanation=Explanation(detector_summary=explanation.detector_summary,
                                no_evidence_notice=explanation.no_evidence_notice),
        limitations=explanation.limitations,
        uncertainty_warning=UNCERTAINTY_WARNING,
    )


def analyze_video_bytes(data: bytes, filename: str, kb: KnowledgeService) -> Report:
    check = validation.validate_video_bytes(data, filename)
    if not check.ok:
        raise ApiError(422, "invalid_file", check.reason or "The video failed validation.")

    temp = validation.safe_temp_path(Path(filename).suffix.lower())
    frames_dir = config.TMP_UPLOAD_DIR / f"frames_{temp.stem}"
    try:
        temp.write_bytes(data)
        meta = validate_video_file(temp)
        if not meta.ok:
            raise ApiError(422, "invalid_video", meta.reason or "The video could not be read.")
        with _inference_slots:
            try:
                result = analyze_video(temp, workdir=frames_dir)
            except DetectorError as exc:
                raise ApiError(503, "detector_unavailable", str(exc)) from exc
    finally:
        validation.cleanup_temp_file(temp)
        shutil.rmtree(frames_dir, ignore_errors=True)

    first = result.frame_results[0].detection
    summary = DetectionResult(
        scores={"ai": result.mean_ai_score, "real": result.mean_real_score},
        category=result.category, model_name=first.model_name,
        model_version_label=first.model_version_label,
        decision_policy=f"Mean-score aggregation across {result.frames_analyzed} frames; {result.aggregation_method}",
        raw_top_label="ai" if result.mean_ai_score >= result.mean_real_score else "real",
        inference_latency_ms=sum(f.detection.inference_latency_ms for f in result.frame_results),
    )
    query = (f"{result.category.value} video frame-level analysis temporal deepfake detection "
             "limitations inconsistent frame scores")
    evidence, passages = _evidence(kb, query)
    explanation = generate_explanation(summary, passages, query_context=result.category.value)
    limitations = list(explanation.limitations) + [
        "Video-specific: this is frame-level analysis, not a validated temporal/video-level detector. "
        "Frame-score variance alone is not evidence of manipulation."
    ]

    return Report(
        id=uuid.uuid4().hex,
        created_at=datetime.now(timezone.utc).isoformat(),
        kind="video",
        app_version=APP_VERSION,
        file=FileInfo(
            name=Path(filename).name[:200], mime=check.detected_mime or "video/*", size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(), duration_seconds=result.video_duration_seconds,
        ),
        detector=DetectorInfo(name=first.model_name, version_label=first.model_version_label,
                              thresholds=_thresholds()),
        result=ModelResult(
            category=result.category.value, predicted_class=summary.raw_top_label, scores=summary.scores,
            decision_policy=summary.decision_policy, latency_ms=summary.inference_latency_ms,
        ),
        video=VideoInfo(
            frames_analyzed=result.frames_analyzed, aggregation_method=result.aggregation_method,
            mean_ai_score=result.mean_ai_score, mean_real_score=result.mean_real_score,
            frames=[FrameInfo(index=f.frame_index, timestamp_seconds=f.timestamp_seconds,
                              predicted_class=f.detection.raw_top_label, scores=f.detection.scores)
                    for f in result.frame_results],
        ),
        analysis_method=("Evenly-spaced frame sampling, per-frame image classification and mean aggregation "
                         "(experimental; not a validated video-level or temporal detector)"),
        evidence=evidence,
        explanation=Explanation(detector_summary=explanation.detector_summary,
                                no_evidence_notice=explanation.no_evidence_notice),
        limitations=limitations,
        uncertainty_warning=UNCERTAINTY_WARNING,
    )


def report_to_markdown(r: Report) -> str:
    lines = ["# VerityAI Analysis Report", "", f"*Generated {r.created_at} · report {r.id}*", "",
             "> " + r.uncertainty_warning, "", "## File",
             f"- **Name:** {r.file.name}", f"- **Type:** {r.file.mime}", f"- **Size:** {r.file.size_bytes} bytes",
             f"- **SHA-256:** {r.file.sha256}"]
    if r.file.width:
        lines.append(f"- **Dimensions:** {r.file.width}x{r.file.height}px")
    if r.file.duration_seconds is not None:
        lines.append(f"- **Duration:** {r.file.duration_seconds:.1f}s")
    lines += ["", "## Model prediction (detector output)",
              f"- **Detector:** {r.detector.version_label}",
              f"- **Category (policy decision):** {r.result.category}",
              f"- **Top class:** {r.result.predicted_class}",
              "- **Raw scores:** " + ", ".join(f"{k}={v:.4f}" for k, v in r.result.scores.items()),
              f"- **Decision policy:** {r.result.decision_policy}",
              f"- **Score semantics:** {r.result.score_semantics}",
              f"- **Method:** {r.analysis_method}"]
    if r.video:
        lines += ["", "### Frames", "| # | t (s) | class | AI | Real |", "|---|---|---|---|---|"]
        for f in r.video.frames:
            lines.append(f"| {f.index} | {f.timestamp_seconds:.2f} | {f.predicted_class} | "
                         f"{f.scores.get('ai', 0):.3f} | {f.scores.get('real', 0):.3f} |")
    lines += ["", "## Retrieved research (separate from the model prediction)"]
    if r.evidence.passages:
        for p in r.evidence.passages:
            loc = f", p.{p.page_number}" if p.page_number else ""
            lines.append(f"{p.citation_id}. **{p.title}**{loc} ({p.source_file})")
            lines.append(f"   > {' '.join(p.text.split())[:600]}")
    else:
        lines.append(f"*{r.explanation.no_evidence_notice or 'No relevant passage was retrieved.'}*")
    lines += ["", "## Limitations"] + [f"- {x}" for x in r.limitations] + [""]
    return "\n".join(lines)
