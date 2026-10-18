"""Tests for report generation (src/reporting.py) and the generation module's
evidence/no-evidence branches (src/rag/generation.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import reporting
from src.detector.base import DetectionResult, ResultCategory
from src.rag.generation import Citation, GeneratedExplanation, generate_explanation
from src.rag.retrieval import EvidencePassage


def _make_detection(category=ResultCategory.AI_GENERATED_SUSPECTED) -> DetectionResult:
    return DetectionResult(
        scores={"ai": 0.91, "real": 0.09},
        category=category,
        model_name="Nahrawy/AIorNot",
        model_version_label="Nahrawy/AIorNot (swin-tiny fine-tune)",
        decision_policy="If P(AI) >= 0.65 -> AI-generated suspected. ...",
        raw_top_label="ai",
        inference_latency_ms=42.0,
    )


class TestGenerationGrounding:
    def test_missing_evidence_produces_explicit_notice(self):
        explanation = generate_explanation(_make_detection(), evidence=[], query_context="test query")
        assert explanation.no_evidence_notice is not None
        assert explanation.research_findings == []
        assert len(explanation.limitations) > 0  # standard limitations always present

    def test_evidence_produces_cited_findings(self):
        passage = EvidencePassage(
            chunk_id="x", text="Detectors trained on GANs underperform on diffusion models.",
            title="Corvi et al. 2023", source_file="corvi2023.md", page_number=None,
            source_url="https://arxiv.org/abs/2211.00680", vector_rank=1, bm25_rank=1, rrf_score=0.03,
        )
        explanation = generate_explanation(_make_detection(), evidence=[passage], query_context="q")
        assert explanation.no_evidence_notice is None
        assert len(explanation.research_findings) == 1
        assert "Corvi et al. 2023" in explanation.research_findings[0]
        assert explanation.citations[0].source_url == "https://arxiv.org/abs/2211.00680"

    def test_detector_summary_never_claims_calibrated_probability(self):
        explanation = generate_explanation(_make_detection(), evidence=[], query_context="q")
        assert "calibrated probability" not in explanation.detector_summary.lower()
        assert "raw scores" in explanation.detector_summary.lower()


class TestImageReport:
    def test_build_and_serialize_json(self):
        explanation = GeneratedExplanation(
            detector_summary="summary text",
            research_findings=["According to X: \"evidence\""],
            citations=[Citation("X", "x.pdf", 3, "https://example.com")],
            limitations=["limitation one"],
            no_evidence_notice=None,
        )
        report = reporting.build_image_report(
            file_name="photo.jpg", file_type="image/jpeg", file_size_bytes=12345,
            image_width=800, image_height=600, detection=_make_detection(), explanation=explanation,
        )
        payload = json.loads(reporting.to_json(report))

        required_fields = {
            "file_name", "file_type", "file_size_bytes", "detector_name", "detector_version",
            "predicted_class", "model_scores", "decision_policy", "predicted_category",
            "analysis_method", "citations", "limitations", "uncertainty_warning",
            "analysis_timestamp_utc",
        }
        assert required_fields.issubset(payload.keys())
        assert payload["predicted_category"] == "AI-generated suspected"
        assert payload["file_size_bytes"] == 12345

    def test_markdown_report_contains_uncertainty_warning(self):
        explanation = GeneratedExplanation("summary", [], [], ["lim"], "no evidence notice")
        report = reporting.build_image_report(
            file_name="photo.jpg", file_type="image/jpeg", file_size_bytes=100,
            image_width=10, image_height=10, detection=_make_detection(ResultCategory.INCONCLUSIVE),
            explanation=explanation,
        )
        md = reporting.to_markdown(report)
        assert "does not prove" in md.lower() or "screening signal only" in md.lower()
        assert "Inconclusive" in md

    def test_report_never_overclaims_authenticity_proof(self):
        """Regression guard: the uncertainty warning must always be present
        and must explicitly disclaim proof of authenticity."""
        explanation = GeneratedExplanation("s", [], [], [], None)
        report = reporting.build_image_report(
            file_name="f.png", file_type="image/png", file_size_bytes=1, image_width=1, image_height=1,
            detection=_make_detection(ResultCategory.REAL_SUSPECTED), explanation=explanation,
        )
        assert "does not prove" in report.uncertainty_warning.lower()
