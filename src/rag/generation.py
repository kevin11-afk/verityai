"""
Evidence-grounded explanation generation using deterministic templates.

No LLM is used here and none is required to run the app. Every sentence
either (a) states a structural fact about the detector's own output, or
(b) directly attributes a claim to a specific retrieved passage with its
citation. The two are never blended into one unattributed sentence, which
is what keeps "what the model said" and "what the research says" separate,
per the project's core scientific requirement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.detector.base import DetectionResult, ResultCategory
from src.rag.retrieval import EvidencePassage


@dataclass
class Citation:
    title: str
    source_file: str
    page_number: Optional[int]
    source_url: Optional[str]

    def format(self) -> str:
        loc = f", p.{self.page_number}" if self.page_number else ""
        return f"{self.title}{loc} ({self.source_file})"


@dataclass
class GeneratedExplanation:
    detector_summary: str
    research_findings: List[str]
    citations: List[Citation]
    limitations: List[str]
    no_evidence_notice: Optional[str]


_STANDARD_LIMITATIONS = [
    "This is an educational screening tool, not a forensic certification service: "
    "it cannot prove whether a file is authentic or establish provenance.",
    "A raw model score is not a calibrated probability of authenticity — it reflects "
    "how similar the image is to the patterns the detector learned from its training data.",
    "Detectors trained on one generation of image/video generators often perform worse on "
    "newer, unseen generators. A low AI score does not guarantee the content is real.",
    "Compression, resizing, screenshots, re-encoding (e.g. social-media recompression), "
    "and routine editing can all shift a detector's score in either direction.",
    "Results on mixed real+generated content (e.g. a real photo with an AI-edited region) "
    "are not well defined by a whole-image classifier.",
]


def describe_detector_output(result: DetectionResult) -> str:
    score_str = ", ".join(f"{label.upper()}={score:.3f}" for label, score in result.scores.items())
    return (
        f"The detector ({result.model_version_label}) classified this image as "
        f"'{result.raw_top_label}' with raw scores [{score_str}], computed in "
        f"{result.inference_latency_ms:.0f}ms. Applying this project's fixed decision "
        f"policy ({result.decision_policy}) yields the category "
        f"'{result.category.value}'."
    )


def generate_explanation(
    detection: DetectionResult, evidence: List[EvidencePassage], query_context: str
) -> GeneratedExplanation:
    detector_summary = describe_detector_output(detection)

    if not evidence:
        return GeneratedExplanation(
            detector_summary=detector_summary,
            research_findings=[],
            citations=[],
            limitations=list(_STANDARD_LIMITATIONS),
            no_evidence_notice=(
                "No retrieved research passage was relevant enough to cite for "
                f"'{query_context}'. The knowledge base may not cover this specific "
                "question yet — see the Research Assistant page to search it directly, "
                "or add more documents via scripts/ingest_documents.py."
            ),
        )

    findings = []
    citations = []
    for passage in evidence:
        snippet = passage.text.strip().replace("\n", " ")
        if len(snippet) > 320:
            snippet = snippet[:320].rsplit(" ", 1)[0] + "…"
        citation = Citation(passage.title, passage.source_file, passage.page_number, passage.source_url)
        citations.append(citation)
        findings.append(f"According to {citation.format()}: \"{snippet}\"")

    return GeneratedExplanation(
        detector_summary=detector_summary,
        research_findings=findings,
        citations=citations,
        limitations=list(_STANDARD_LIMITATIONS),
        no_evidence_notice=None,
    )


def answer_research_question(question: str, evidence: List[EvidencePassage]) -> GeneratedExplanation:
    """Research Assistant page: same grounding rules, no detector output."""
    if not evidence:
        return GeneratedExplanation(
            detector_summary="",
            research_findings=[],
            citations=[],
            limitations=[],
            no_evidence_notice=(
                f"The knowledge base does not contain a passage relevant enough to "
                f"answer \"{question}\". Try rephrasing, or add more source documents "
                f"to knowledge_base/ and re-run scripts/ingest_documents.py."
            ),
        )

    findings, citations = [], []
    for passage in evidence:
        snippet = passage.text.strip().replace("\n", " ")
        if len(snippet) > 500:
            snippet = snippet[:500].rsplit(" ", 1)[0] + "…"
        citation = Citation(passage.title, passage.source_file, passage.page_number, passage.source_url)
        citations.append(citation)
        findings.append(f"According to {citation.format()}: \"{snippet}\"")

    return GeneratedExplanation(
        detector_summary="",
        research_findings=findings,
        citations=citations,
        limitations=[],
        no_evidence_notice=None,
    )
