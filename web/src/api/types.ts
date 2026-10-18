// Mirrors api/schemas.py. Keep the two in sync.

export interface ApiErrorBody { code: string; message: string }

export interface FileInfo {
  name: string; mime: string; size_bytes: number; sha256: string;
  width?: number | null; height?: number | null; duration_seconds?: number | null;
}
export interface Thresholds { ai: number; real: number }
export interface DetectorInfo { name: string; version_label: string; thresholds: Thresholds }
export interface ModelResult {
  category: string; predicted_class: string; scores: Record<string, number>;
  decision_policy: string; latency_ms: number; score_semantics: string;
}
export interface FrameInfo { index: number; timestamp_seconds: number; predicted_class: string; scores: Record<string, number> }
export interface VideoInfo {
  frames_analyzed: number; aggregation_method: string; mean_ai_score: number; mean_real_score: number; frames: FrameInfo[];
}
export interface Passage {
  citation_id: number; chunk_id: string; title: string; source_file: string; page_number?: number | null;
  source_url?: string | null; text: string; vector_rank?: number | null; bm25_rank?: number | null;
  rrf_score: number; vector_distance?: number | null;
}
export interface RetrievalDiag {
  query: string; corpus_chunks: number; best_vector_distance?: number | null; distance_threshold: number;
  abstained: boolean; abstain_reason?: string | null; gate_passed_by?: "vector" | "lexical" | null;
  lexical_coverage?: number | null; vector_candidates: number; bm25_candidates: number; top_k: number; weak_filtered?: number;
}
export interface Evidence { passages: Passage[]; diagnostics: RetrievalDiag }
export interface Explanation { detector_summary: string; no_evidence_notice?: string | null }
export interface Report {
  id: string; created_at: string; kind: "image" | "video"; app_version: string;
  file: FileInfo; detector: DetectorInfo; result: ModelResult; video?: VideoInfo | null;
  analysis_method: string; evidence: Evidence; explanation: Explanation; limitations: string[]; uncertainty_warning: string;
}
export interface ReportSummary {
  id: string; created_at: string; kind: "image" | "video"; file_name: string; size_bytes: number;
  category: string; detector_version: string;
}
export interface ReportList { reports: ReportSummary[]; total: number; persistence_mode: string; persistence_note: string }

export interface Claim { text: string; citation_id: number }
export interface AskResponse {
  question: string; abstained: boolean; notice?: string | null; answer: Claim[]; passages: Passage[]; diagnostics: RetrievalDiag;
}

export interface KBDocument {
  source_file: string; title: string; origin: "seed" | "upload"; removable: boolean; chunks: number; indexed: boolean;
  sha256?: string | null; size_bytes?: number | null; source_url?: string | null; duplicate_of?: string | null;
}
export interface KBList { documents: KBDocument[]; total_chunks: number; persistence_mode: string; persistence_note: string }
export interface KBUploadResult { status: "indexed" | "duplicate"; message: string; chunks_added: number; document?: KBDocument | null }
export interface KBActionResult { message: string; chunks: number }

export interface ComponentStatus { name: string; loaded: boolean; detail?: string | null }
export interface Limits {
  max_image_mb: number; max_video_mb: number; max_video_seconds: number; max_video_frames: number;
  max_kb_upload_mb: number; image_extensions: string[]; video_extensions: string[]; document_extensions: string[];
}
export interface Health {
  status: "ok" | "degraded"; version: string; python: string; detector: ComponentStatus; detector_version_label: string;
  thresholds: Thresholds; embedder: ComponentStatus; knowledge_base_documents: number; knowledge_base_chunks: number;
  knowledge_base_ready: boolean; persistence_mode: string; persistence_note: string; limits: Limits; rss_mb?: number | null;
}

// ---- evaluation artifacts (scripts/export_evaluation.py) -------------------
export type Ci = [number, number] | null;
export interface CalibrationBin { lo: number; hi: number; n: number; mean_score: number; fraction_ai: number }
export interface GroupStat { name: string; label: number; n: number; flagged_ai: number; flagged_rate: number; flagged_ci: Ci; mean_ai_score: number }
export interface PolicyOutcomes {
  ai_suspected: number; real_suspected: number; inconclusive: number; ai_suspected_correct: number; real_suspected_correct: number;
}
export interface PolicyStats {
  threshold: number; tp: number; fp: number; tn: number; fn: number; accuracy: number; accuracy_ci: Ci;
  ai_recall: number; ai_recall_ci: Ci; real_recall: number; real_recall_ci: Ci; policy_outcomes: PolicyOutcomes;
}
export interface DetectorVariant {
  auc: number; auc_ci: [number, number]; average_precision: number;
  roc: [number, number, number | null][]; pr: [number, number, number | null][];
  calibration: { bins: CalibrationBin[]; ece: number; brier: number; valid: boolean; note: string };
  at_policy_threshold: PolicyStats; groups: GroupStat[]; samples: { labels: number[]; scores: number[] };
}
export interface DetectorModelEval {
  id: string; license: string; commercial_use: boolean; note: string; is_production: boolean;
  variants: Record<string, DetectorVariant>;
}
export interface DetectorEval {
  schema_version: number; generated_at: string; git_commit: string;
  policy: { ai_threshold: number; real_threshold: number; production_model: string };
  dataset: {
    name: string; split: string; sampling: string; n_real: number; n_ai: number; license_note: string;
    variants: Record<string, string>; limitations: string[];
  };
  models: DetectorModelEval[];
}
export interface Rate { k: number; n: number; rate: number; ci: Ci }
export interface RagRow {
  id: string; category: "answerable" | "unanswerable_in_domain" | "off_topic"; question: string; expected_sources: string[];
  retrieved_sources: string[]; abstained: boolean; best_vector_distance: number | null; first_rank: number | null;
  n_claims: number; claims_grounded: boolean;
}
export interface RagEval {
  schema_version: number; generated_at: string; git_commit: string;
  config: { top_k: number; distance_threshold: number; embedding_model: string; chunks: number; documents: number };
  question_set: { answerable: number; unanswerable_in_domain: number; off_topic: number; labeling: string };
  retrieval: {
    hit_at_1: Rate; hit_at_3: Rate; hit_at_5: Rate; mrr: number; mrr_ci: [number, number];
    citation_coverage: { k: number; n: number }; wrongly_abstained: string[];
  };
  ablation: { mode: string; hit_at_1: number; hit_at_5: number; mrr: number }[];
  abstention: Record<string, { n: number; abstained: number; rate: number; ci: Ci }>;
  abstention_sweep: { threshold: number; answerable_answered: number; answerable_total: number; refusals_correct: number; refusals_total: number }[];
  groundedness: { claims_total: number; answers_fully_grounded: number; answers: number; method: string };
  rows: RagRow[];
}
export interface EvaluationResponse { detector: DetectorEval | null; rag: RagEval | null }
