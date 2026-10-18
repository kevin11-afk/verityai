"""
Real pretrained AI-generated-image detector.

Wraps a Hugging Face `transformers` image-classification pipeline
(Nahrawy/AIorNot by default — see src/config.py for why it was chosen).
This module NEVER returns a random or hardcoded prediction: if the model
cannot be loaded or inference fails, it raises DetectorError so the caller
can show a clear error instead of a fabricated result.

The model is loaded lazily and cached process-wide (via functools.lru_cache)
so repeated calls in the same Streamlit session do not reload ~110MB of
weights from disk every time, and so the app never tries to hold two large
models in memory at once.
"""
from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Union

from src import config
from src.detector.base import DetectionResult, DetectorError, ResultCategory


@lru_cache(maxsize=1)
def _load_pipeline():
    """Load the HF image-classification pipeline once per process.

    Imports are deliberately local to this function (lazy import) so that
    importing src.detector.image_detector elsewhere in the app does not pay
    the multi-second torch/transformers import cost until a prediction is
    actually requested.
    """
    try:
        import torch
        from transformers import pipeline
    except ImportError as exc:
        raise DetectorError(
            "Required packages (torch, transformers) are not installed. "
            "Run: pip install -r requirements.txt"
        ) from exc

    try:
        device = -1  # CPU; see README for why GPU is not assumed on the free hosting tier
        clf = pipeline(
            "image-classification",
            model=config.DETECTOR_MODEL_NAME,
            device=device,
        )
        return clf
    except Exception as exc:  # noqa: BLE001 - surfaced to the user as a clear error
        raise DetectorError(
            f"Could not load detector model '{config.DETECTOR_MODEL_NAME}'. "
            f"This usually means no internet access to Hugging Face Hub, or "
            f"the model name is wrong. Underlying error: {exc}"
        ) from exc


def _decide_category(scores: dict) -> tuple[ResultCategory, str]:
    """Apply the project's fixed, documented decision policy to raw scores.

    The policy is intentionally simple and stated in full so the report can
    quote it verbatim rather than implying a calibrated probability exists.
    """
    ai_score = scores.get("ai") or scores.get("AI") or scores.get("fake") or scores.get("FAKE") or 0.0
    real_score = scores.get("real") or scores.get("REAL") or scores.get("human") or 0.0

    policy = (
        f"If P(AI) >= {config.AI_SUSPECTED_THRESHOLD:.2f} -> 'AI-generated suspected'. "
        f"If P(Real) >= {config.REAL_SUSPECTED_THRESHOLD:.2f} -> 'Real or non-synthetic suspected'. "
        f"Otherwise -> 'Inconclusive'. Scores are the model's raw softmax "
        f"outputs, not a calibrated probability of authenticity."
    )

    if ai_score >= config.AI_SUSPECTED_THRESHOLD:
        return ResultCategory.AI_GENERATED_SUSPECTED, policy
    if real_score >= config.REAL_SUSPECTED_THRESHOLD:
        return ResultCategory.REAL_SUSPECTED, policy
    return ResultCategory.INCONCLUSIVE, policy


def predict_image(image_path: Union[str, Path]) -> DetectionResult:
    """Run the real detector on one image file and return a DetectionResult.

    Raises DetectorError on any failure (model load, corrupt image passed
    the earlier validation step but still unreadable by the model, etc.) —
    callers must not catch this and substitute a fake result.
    """
    clf = _load_pipeline()

    start = time.perf_counter()
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            # Explicit softmax: left to its default, the pipeline applied an
            # independent sigmoid to this model's logits (scores did not sum
            # to 1), contradicting the documented decision policy.
            raw_predictions = clf(img, function_to_apply="softmax")
    except Exception as exc:  # noqa: BLE001
        raise DetectorError(f"Inference failed on '{image_path}': {exc}") from exc
    latency_ms = (time.perf_counter() - start) * 1000

    # raw_predictions is a list like [{"label": "ai", "score": 0.92}, {"label": "real", "score": 0.08}]
    scores = {p["label"].lower(): float(p["score"]) for p in raw_predictions}
    top_label = max(raw_predictions, key=lambda p: p["score"])["label"]

    category, policy = _decide_category(scores)

    return DetectionResult(
        scores=scores,
        category=category,
        model_name=config.DETECTOR_MODEL_NAME,
        model_version_label=config.DETECTOR_VERSION_LABEL,
        decision_policy=policy,
        raw_top_label=top_label,
        inference_latency_ms=latency_ms,
    )


def is_model_available() -> bool:
    """Best-effort check used by the UI to show a friendly message instead
    of a stack trace if the model can't be loaded (e.g. no internet)."""
    try:
        _load_pipeline()
        return True
    except DetectorError:
        return False
