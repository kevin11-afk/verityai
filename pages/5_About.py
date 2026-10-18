"""VerityAI — About / Methodology page."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src import config

st.set_page_config(page_title=f"{config.APP_NAME} — About", page_icon="ℹ️", layout="wide")
st.title("ℹ️ About & Methodology")

st.header("What VerityAI is — and isn't")
st.markdown(
    """
    VerityAI is an **educational screening tool** that combines a real
    pretrained image classifier with Retrieval-Augmented Generation (RAG)
    over AI-media-forensics research. It is **not** a forensic
    certification service, cannot prove a file's authenticity or
    provenance, and should never be the sole basis for a consequential
    decision about a specific piece of media.
    """
)

st.header("Architecture")
st.markdown(
    """
    ```
    Upload (image/video)
        │
        ▼
    Validation (type, size, dimensions/duration, signature check)
        │
        ▼
    Real detector inference (Nahrawy/AIorNot — Swin-Tiny, CPU)
        │                                   │
        │ (category + raw scores)           │ (used to build a retrieval query)
        ▼                                   ▼
    Decision policy (fixed thresholds)   Hybrid RAG retrieval
        │                                   │
        │          BM25 keyword search ─────┤
        │          + ChromaDB vector search ┤── Reciprocal Rank Fusion
        │                                   │
        └──────────────┬────────────────────┘
                        ▼
            Deterministic template-based explanation
            (detector output and research findings kept
             strictly separate, every claim cited)
                        │
                        ▼
                Report (on-screen + JSON/Markdown download)
    ```
    No LLM is used anywhere in this pipeline, and none is required to run
    the app — see "Explanation engine" below.
    """
)

st.header("Model information")
st.markdown(
    f"""
    **Image detector:** `{config.DETECTOR_MODEL_NAME}` — {config.DETECTOR_VERSION_LABEL}
    - Fine-tuned `microsoft/swin-tiny-patch4-window7-224` (~28M parameters), Apache-2.0 license.
    - Trained on the "aiornot" image-classification competition dataset (~62k images, Real vs. AI).
    - Selected over three other candidates for its permissive license and smallest
      footprint — full comparison in `knowledge_base/04_detector_model_cards.md`
      (also answerable from the Research Assistant page).
    - **Known limitation:** the training data predates 2024, so newer generators
      (SDXL-Turbo, Midjourney v6, DALL-E 3, Flux, etc.) are out-of-distribution.
      A low AI score does **not** guarantee the image is real — see
      `knowledge_base/02_generalization_and_dataset_bias.md`.

    **Embedding model:** `{config.EMBEDDING_MODEL_NAME}` — 22.7M parameters,
    384-dimensional embeddings, Apache-2.0 license.

    **Decision policy (fixed, not learned on any of this project's own data):**
    AI score ≥ {config.AI_SUSPECTED_THRESHOLD:.2f} → "AI-generated suspected";
    Real score ≥ {config.REAL_SUSPECTED_THRESHOLD:.2f} → "Real or non-synthetic suspected";
    otherwise → "Inconclusive". A raw softmax score is reported as exactly that —
    never described as a calibrated probability.
    """
)

st.header("Explanation engine (no paid LLM)")
st.markdown(
    """
    Explanations are produced by **deterministic templates**
    (`src/rag/generation.py`), not a generative language model. Every
    sentence either (a) states a structural fact about the detector's own
    output, computed in code, or (b) directly quotes and cites a specific
    retrieved passage. The two are never blended into one unattributed
    sentence — this is what "evidence-grounded" means in this project, and
    it is enforced in code, not just in a prompt.
    """
)

st.header("Ethical use")
st.markdown(
    """
    - Do not use VerityAI's output as the sole basis for accusing a
      specific person of creating or sharing fabricated media — a
      screening signal is not proof.
    - Do not use this tool, or any detector, to help a generator evade
      detection; several of the underlying model cards explicitly
      prohibit that use.
    - Treat "Inconclusive" results as genuinely inconclusive, not as a
      soft "probably real."
    """
)

st.header("Limitations (full list)")
st.markdown(
    """
    - Image detector scope is a single pretrained binary classifier; it is
      not an ensemble and has not been independently re-evaluated by this
      project on a fresh, out-of-distribution test set (see Evaluation
      Dashboard — real numbers only, "Not evaluated" otherwise).
    - Video analysis is frame-level only — not a temporal/video-level
      detector. See `knowledge_base/03_video_and_deepfake_detection.md`.
    - The knowledge base is a small, hand-curated starter set (6
      documents). It will say so explicitly ("insufficient evidence")
      rather than guess when a question falls outside it.
    - Compression, resizing, screenshots, and routine editing can shift
      detector scores in either direction.
    - Mixed real+AI content (e.g. a real photo with an AI-edited region)
      is not well modeled by a whole-image classifier.
    """
)

st.header("Privacy")
st.markdown(
    """
    - Uploaded images/videos are processed locally (wherever this Streamlit
      process is running) and written only to a temporary folder
      (`data/tmp/`), deleted immediately after analysis — see
      `src/validation.py: cleanup_temp_file`.
    - No uploaded media, filename, or analysis result is sent to any
      external API. The only network calls this app makes are: (a) the
      one-time download of the pretrained models from Hugging Face Hub on
      first run (cached locally afterward), and (b) nothing else.
    - No analytics, tracking, or logging of uploaded file contents.
    """
)

st.header("Credits")
st.markdown(
    """
    Built as a student AI/Data-Science portfolio project. Detector and
    embedding models are used under their original authors' licenses
    (Apache-2.0 for both defaults) — see `NOTICE.md`. Research citations
    throughout the knowledge base link to their original, independently
    verifiable sources.
    """
)
