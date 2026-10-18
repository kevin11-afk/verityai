"""
Assembles the full analysis report (image or video) and serializes it to
JSON or Markdown for download. This is the one place that defines what a
"report" contains, so every required field from the project spec is
guaranteed present — nothing here is optional silently.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import List, Optional

from src.detector.base import DetectionResult
from src.rag.generation import GeneratedExplanation


@dataclass
class ImageReport:
    file_name: str
    file_type: str
    file_size_bytes: int
    image_width: int
    image_height: int
    detector_name: str
    detector_version: str
    predicted_class: str
    model_scores: dict
    decision_policy: str
    predicted_category: str
    analysis_method: str
    retrieved_evidence_count: int
    citations: List[str]
    explanation_detector_summary: str
    explanation_research_findings: List[str]
    no_evidence_notice: Optional[str]
    limitations: List[str]
    uncertainty_warning: str
    analysis_timestamp_utc: str


@dataclass
class VideoReport:
    file_name: str
    file_type: str
    file_size_bytes: int
    video_duration_seconds: float
    detector_name: str
    detector_version: str
    frames_analyzed: int
    frame_level_scores: List[dict]
    aggregation_method: str
    mean_ai_score: float
    mean_real_score: float
    predicted_category: str
    analysis_method: str
    retrieved_evidence_count: int
    citations: List[str]
    explanation_detector_summary: str
    explanation_research_findings: List[str]
    no_evidence_notice: Optional[str]
    limitations: List[str]
    uncertainty_warning: str
    analysis_timestamp_utc: str


_UNCERTAINTY_WARNING = (
    "This result is an automated screening signal only. It does not prove whether "
    "the file is authentic, AI-generated, or edited, and must not be treated as a "
    "forensic or legal determination. Always corroborate with other evidence."
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_image_report(
    *,
    file_name: str,
    file_type: str,
    file_size_bytes: int,
    image_width: int,
    image_height: int,
    detection: DetectionResult,
    explanation: GeneratedExplanation,
) -> ImageReport:
    return ImageReport(
        file_name=file_name,
        file_type=file_type,
        file_size_bytes=file_size_bytes,
        image_width=image_width,
        image_height=image_height,
        detector_name=detection.model_name,
        detector_version=detection.model_version_label,
        predicted_class=detection.raw_top_label,
        model_scores=detection.scores,
        decision_policy=detection.decision_policy,
        predicted_category=detection.category.value,
        analysis_method="Single-pass image classification with a pretrained Swin-Tiny model",
        retrieved_evidence_count=len(explanation.citations),
        citations=[c.format() for c in explanation.citations],
        explanation_detector_summary=explanation.detector_summary,
        explanation_research_findings=explanation.research_findings,
        no_evidence_notice=explanation.no_evidence_notice,
        limitations=explanation.limitations,
        uncertainty_warning=_UNCERTAINTY_WARNING,
        analysis_timestamp_utc=_utc_now_iso(),
    )


def build_video_report(
    *,
    file_name: str,
    file_type: str,
    file_size_bytes: int,
    video_duration_seconds: float,
    frames_analyzed: int,
    frame_level_scores: List[dict],
    aggregation_method: str,
    mean_ai_score: float,
    mean_real_score: float,
    predicted_category: str,
    detector_name: str,
    detector_version: str,
    explanation: GeneratedExplanation,
) -> VideoReport:
    return VideoReport(
        file_name=file_name,
        file_type=file_type,
        file_size_bytes=file_size_bytes,
        video_duration_seconds=video_duration_seconds,
        detector_name=detector_name,
        detector_version=detector_version,
        frames_analyzed=frames_analyzed,
        frame_level_scores=frame_level_scores,
        aggregation_method=aggregation_method,
        mean_ai_score=mean_ai_score,
        mean_real_score=mean_real_score,
        predicted_category=predicted_category,
        analysis_method="Evenly-spaced frame sampling + per-frame image classification + mean aggregation "
                         "(experimental; not a validated video-level/temporal detector)",
        retrieved_evidence_count=len(explanation.citations),
        citations=[c.format() for c in explanation.citations],
        explanation_detector_summary=explanation.detector_summary,
        explanation_research_findings=explanation.research_findings,
        no_evidence_notice=explanation.no_evidence_notice,
        limitations=explanation.limitations,
        uncertainty_warning=_UNCERTAINTY_WARNING,
        analysis_timestamp_utc=_utc_now_iso(),
    )


def to_json(report) -> str:
    return json.dumps(asdict(report), indent=2, ensure_ascii=False)


def to_markdown(report) -> str:
    data = asdict(report)
    lines = [f"# VerityAI Analysis Report", ""]
    lines.append(f"*Generated {data['analysis_timestamp_utc']}*")
    lines.append("")
    lines.append("> " + _UNCERTAINTY_WARNING)
    lines.append("")
    lines.append("## File")
    for key in ("file_name", "file_type", "file_size_bytes"):
        if key in data:
            lines.append(f"- **{key.replace('_', ' ').title()}:** {data[key]}")
    if "image_width" in data:
        lines.append(f"- **Dimensions:** {data['image_width']}x{data['image_height']}px")
    if "video_duration_seconds" in data:
        lines.append(f"- **Duration:** {data['video_duration_seconds']:.1f}s")
    lines.append("")

    lines.append("## Detection")
    lines.append(f"- **Detector:** {data['detector_name']} ({data['detector_version']})")
    if "predicted_class" in data:
        lines.append(f"- **Predicted class:** {data['predicted_class']}")
        lines.append(f"- **Model scores:** {data['model_scores']}")
    if "frames_analyzed" in data:
        lines.append(f"- **Frames analyzed:** {data['frames_analyzed']}")
        lines.append(f"- **Aggregation method:** {data['aggregation_method']}")
        lines.append(f"- **Mean AI score:** {data['mean_ai_score']:.3f}")
        lines.append(f"- **Mean Real score:** {data['mean_real_score']:.3f}")
    lines.append(f"- **Decision policy:** {data.get('decision_policy', '(see aggregation method)')}" )
    lines.append(f"- **Predicted category:** {data['predicted_category']}")
    lines.append(f"- **Analysis method:** {data['analysis_method']}")
    lines.append("")

    lines.append("## Explanation")
    if data.get("explanation_detector_summary"):
        lines.append(data["explanation_detector_summary"])
        lines.append("")
    if data["explanation_research_findings"]:
        lines.append("**Research evidence (retrieved from the knowledge base):**")
        for finding in data["explanation_research_findings"]:
            lines.append(f"- {finding}")
    elif data.get("no_evidence_notice"):
        lines.append(f"*{data['no_evidence_notice']}*")
    lines.append("")

    if data["citations"]:
        lines.append("## Citations")
        for c in data["citations"]:
            lines.append(f"- {c}")
        lines.append("")

    lines.append("## Limitations")
    for lim in data["limitations"]:
        lines.append(f"- {lim}")
    lines.append("")

    return "\n".join(lines)
