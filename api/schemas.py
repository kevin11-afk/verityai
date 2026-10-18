"""Typed API contracts. The TypeScript types in web/src/api/types.ts mirror these."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


# ---- analysis / reports ---------------------------------------------------
class FileInfo(BaseModel):
    name: str
    mime: str
    size_bytes: int
    sha256: str
    width: Optional[int] = None
    height: Optional[int] = None
    duration_seconds: Optional[float] = None


class Thresholds(BaseModel):
    ai: float
    real: float


class DetectorInfo(BaseModel):
    name: str
    version_label: str
    thresholds: Thresholds


class ModelResult(BaseModel):
    category: str
    predicted_class: str
    scores: Dict[str, float]
    decision_policy: str
    latency_ms: float
    score_semantics: str = (
        "Raw softmax outputs of the classifier. They are not calibrated probabilities "
        "of authenticity."
    )


class FrameInfo(BaseModel):
    index: int
    timestamp_seconds: float
    predicted_class: str
    scores: Dict[str, float]


class VideoInfo(BaseModel):
    frames_analyzed: int
    aggregation_method: str
    mean_ai_score: float
    mean_real_score: float
    frames: List[FrameInfo]


class Passage(BaseModel):
    citation_id: int = Field(description="1-based marker used by answer claims and report citations")
    chunk_id: str
    title: str
    source_file: str
    page_number: Optional[int] = None
    source_url: Optional[str] = None
    text: str
    vector_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    rrf_score: float
    vector_distance: Optional[float] = None


class RetrievalDiag(BaseModel):
    query: str
    corpus_chunks: int
    best_vector_distance: Optional[float] = None
    distance_threshold: float
    abstained: bool
    abstain_reason: Optional[str] = None
    gate_passed_by: Optional[Literal["vector", "lexical"]] = None
    lexical_coverage: Optional[float] = None
    vector_candidates: int
    bm25_candidates: int
    top_k: int
    weak_filtered: int = 0


class Evidence(BaseModel):
    passages: List[Passage]
    diagnostics: RetrievalDiag


class Explanation(BaseModel):
    detector_summary: str
    no_evidence_notice: Optional[str] = None


class Report(BaseModel):
    id: str
    created_at: str
    kind: Literal["image", "video"]
    app_version: str
    file: FileInfo
    detector: DetectorInfo
    result: ModelResult
    video: Optional[VideoInfo] = None
    analysis_method: str
    evidence: Evidence
    explanation: Explanation
    limitations: List[str]
    uncertainty_warning: str


class ReportSummary(BaseModel):
    id: str
    created_at: str
    kind: Literal["image", "video"]
    file_name: str
    size_bytes: int
    category: str
    detector_version: str


class ReportList(BaseModel):
    reports: List[ReportSummary]
    total: int
    persistence_mode: str
    persistence_note: str


# ---- research ---------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)


class Claim(BaseModel):
    text: str = Field(description="A sentence quoted verbatim (after Markdown cleanup) from a retrieved passage")
    citation_id: int


class AskResponse(BaseModel):
    question: str
    abstained: bool
    notice: Optional[str] = None
    answer: List[Claim]
    passages: List[Passage]
    diagnostics: RetrievalDiag


# ---- knowledge base ----------------------------------------------------------
class KBDocument(BaseModel):
    source_file: str
    title: str
    origin: Literal["seed", "upload"]
    removable: bool
    chunks: int
    indexed: bool
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    source_url: Optional[str] = None
    duplicate_of: Optional[str] = None


class KBList(BaseModel):
    documents: List[KBDocument]
    total_chunks: int
    persistence_mode: str
    persistence_note: str


class KBUploadResult(BaseModel):
    status: Literal["indexed", "duplicate"]
    message: str
    chunks_added: int
    document: Optional[KBDocument] = None


class KBActionResult(BaseModel):
    message: str
    chunks: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)


# ---- health -------------------------------------------------------------------
class ComponentStatus(BaseModel):
    name: str
    loaded: bool
    detail: Optional[str] = None


class Limits(BaseModel):
    max_image_mb: int
    max_video_mb: int
    max_video_seconds: int
    max_video_frames: int
    max_kb_upload_mb: int
    image_extensions: List[str]
    video_extensions: List[str]
    document_extensions: List[str]


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    python: str
    detector: ComponentStatus
    detector_version_label: str
    thresholds: Thresholds
    embedder: ComponentStatus
    knowledge_base_documents: int
    knowledge_base_chunks: int
    knowledge_base_ready: bool
    persistence_mode: str
    persistence_note: str
    limits: Limits
    rss_mb: Optional[float] = None


class EvaluationResponse(BaseModel):
    detector: Optional[Dict[str, Any]] = None
    rag: Optional[Dict[str, Any]] = None
