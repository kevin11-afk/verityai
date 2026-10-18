"""
Central configuration for VerityAI.

Single source of truth for model names, size limits, and thresholds so the
rest of the codebase never hardcodes a magic number. Everything here can be
overridden with an environment variable of the same name (useful for testing
or for tuning on a specific host without editing code).
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
DATA_DIR = PROJECT_ROOT / "data"
TMP_UPLOAD_DIR = DATA_DIR / "tmp"

# --------------------------------------------------------------------------
# Detector model
# --------------------------------------------------------------------------
# Nahrawy/AIorNot: a Swin-Tiny (microsoft/swin-tiny-patch4-window7-224)
# fine-tuned on the "aiornot" image-classification competition dataset.
# Chosen over alternatives (umm-maybe/AI-image-detector, Organika/sdxl-detector,
# dima806/ai_vs_real_image_detection) because:
#   - Apache-2.0 license (fully permissive; the others are CC-BY-ND or
#     CC-BY-NC, or trained on a narrow/low-resolution benchmark).
#   - Swin-Tiny is ~28M parameters (~110MB fp32) — the smallest of the
#     candidates, which matters for an 8GB-RAM laptop and a ~1GB-RAM free
#     hosting tier.
#   - Trained on real photographs vs. generated images from a mixed set of
#     generators (not a single-generator or low-resolution synthetic
#     benchmark like CIFAKE), which generalizes a little better in practice.
# Documented limitation (see About page / report "limitations"): the
# training data predates 2024, so newer generators (SDXL, Midjourney v6,
# DALL-E 3, Flux, etc.) are out-of-distribution and may be scored less
# reliably. This is disclosed to the user in every report, never hidden.
DETECTOR_MODEL_NAME = os.environ.get("VERITYAI_DETECTOR_MODEL", "Nahrawy/AIorNot")
DETECTOR_VERSION_LABEL = "Nahrawy/AIorNot (swin-tiny-patch4-window7-224 fine-tune)"

# Decision policy: the model outputs two softmax scores (REAL, AI). These
# thresholds define the three report categories. They are a documented,
# fixed policy — not learned on this project's own data — so the report
# always states them explicitly rather than implying a calibrated
# probability.
AI_SUSPECTED_THRESHOLD = float(os.environ.get("VERITYAI_AI_THRESHOLD", "0.65"))
REAL_SUSPECTED_THRESHOLD = float(os.environ.get("VERITYAI_REAL_THRESHOLD", "0.65"))
# i.e. score_ai >= 0.65 -> "AI-generated suspected"
#      score_real >= 0.65 -> "Real / non-synthetic suspected"
#      otherwise           -> "Inconclusive"

# --------------------------------------------------------------------------
# Embeddings / RAG
# --------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = os.environ.get(
    "VERITYAI_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
CHROMA_COLLECTION_NAME = "verityai_knowledge_base"
CHUNK_SIZE_CHARS = int(os.environ.get("VERITYAI_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP_CHARS = int(os.environ.get("VERITYAI_CHUNK_OVERLAP", "150"))
RETRIEVAL_TOP_K = int(os.environ.get("VERITYAI_TOP_K", "5"))
# Relevance gate: if even the best vector match is farther than this (cosine
# distance, 0 = identical), the query is treated as outside the knowledge base
# and NO passages are returned, so the app says "insufficient evidence" instead
# of citing unrelated text. Calibrated on 18 queries against the starter KB
# (on-topic best distance <= 0.54, off-topic >= 0.72); re-check if the KB grows.
MAX_BEST_VECTOR_DISTANCE = float(os.environ.get("VERITYAI_MAX_DISTANCE", "0.65"))
RRF_K_CONSTANT = 60  # standard Reciprocal Rank Fusion smoothing constant

# --------------------------------------------------------------------------
# Upload / validation limits
# --------------------------------------------------------------------------
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_SIZE_MB = 15
MAX_IMAGE_DIMENSION_PX = 6000  # reject absurdly large images (decompression-bomb guard)
MIN_IMAGE_DIMENSION_PX = 32

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov"}
ALLOWED_VIDEO_MIME_TYPES = {"video/mp4", "video/quicktime"}
MAX_VIDEO_SIZE_MB = 60
MAX_VIDEO_DURATION_SECONDS = 30
MAX_VIDEO_FRAMES_SAMPLED = 8

ALLOWED_DOC_EXTENSIONS = {".pdf", ".txt", ".md"}

# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------
APP_NAME = "VerityAI"
APP_TAGLINE = "Investigate the media. Examine the evidence."
