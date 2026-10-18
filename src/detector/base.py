"""
Shared types for detector modules.

Keeping the result shape in one place means the Streamlit pages, the
reporting module, and the tests all agree on what a "detection result" is,
and nothing downstream can accidentally invent fields that don't come from
a real model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class ResultCategory(str, Enum):
    AI_GENERATED_SUSPECTED = "AI-generated suspected"
    REAL_SUSPECTED = "Real or non-synthetic suspected"
    INCONCLUSIVE = "Inconclusive"


class DetectorError(RuntimeError):
    """Raised when the model cannot be loaded or inference fails.

    Callers must show this to the user as a clear error (per the project's
    requirement to never pretend an analysis succeeded) rather than
    swallowing it and fabricating a result.
    """


@dataclass
class DetectionResult:
    """The real output of one detector call on one image.

    `scores` are the raw softmax outputs for every class the model was
    trained on, keyed by the model's own label strings — nothing here is
    invented or re-labeled. `category` is this project's own decision-policy
    label (see src/config.py thresholds), computed from `scores`, and is
    reported as a policy decision, not as the model's own output.
    """
    scores: Dict[str, float]
    category: ResultCategory
    model_name: str
    model_version_label: str
    decision_policy: str
    raw_top_label: str
    inference_latency_ms: float
    error: Optional[str] = None
    extra: Dict[str, object] = field(default_factory=dict)
