"""
Experimental video screening: evenly-spaced frame extraction + the SAME
real image detector run per frame, with a documented aggregation method.

This is explicitly NOT a temporal video-deepfake detector. No model here
has been trained or evaluated on video/temporal artifacts. Frame-level
predictions are aggregated with a simple, disclosed rule; inconsistent
frame scores are never presented as proof of manipulation, per the
project's scientific requirements.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

from src import config
from src.detector.base import DetectionResult, DetectorError, ResultCategory
from src.detector.image_detector import predict_image


@dataclass
class VideoValidationResult:
    ok: bool
    reason: str | None = None
    duration_seconds: float | None = None
    frame_count: int | None = None
    fps: float | None = None


@dataclass
class FrameResult:
    frame_index: int
    timestamp_seconds: float
    frame_path: Path
    detection: DetectionResult


@dataclass
class VideoAnalysisResult:
    frame_results: List[FrameResult]
    aggregation_method: str
    mean_ai_score: float
    mean_real_score: float
    category: ResultCategory
    frames_analyzed: int
    video_duration_seconds: float


def validate_video_file(path: Union[str, Path]) -> VideoValidationResult:
    """Check duration using OpenCV now that the file is on disk (the raw-byte
    signature check in src.validation already ran before this)."""
    import cv2  # lazy import; cv2 is already a project dependency

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return VideoValidationResult(False, "Could not open video file; it may be corrupt "
                                             "or use an unsupported codec.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    if fps <= 0 or frame_count <= 0:
        return VideoValidationResult(False, "Video metadata is missing or invalid "
                                             "(zero FPS or frame count).")

    duration_seconds = frame_count / fps
    if duration_seconds > config.MAX_VIDEO_DURATION_SECONDS:
        return VideoValidationResult(
            False,
            f"Video is {duration_seconds:.1f}s, exceeds the "
            f"{config.MAX_VIDEO_DURATION_SECONDS}s limit for this demo.",
            duration_seconds, frame_count, fps,
        )

    return VideoValidationResult(True, None, duration_seconds, frame_count, fps)


def extract_evenly_spaced_frames(
    path: Union[str, Path], out_dir: Path, max_frames: int = config.MAX_VIDEO_FRAMES_SAMPLED
) -> List[tuple[int, float, Path]]:
    """Extract up to `max_frames` evenly spaced frames with OpenCV.

    Returns a list of (frame_index, timestamp_seconds, saved_frame_path).
    """
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise DetectorError(f"Could not reopen video '{path}' for frame extraction.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        cap.release()
        raise DetectorError("Video has zero frames; cannot sample.")

    n_samples = min(max_frames, frame_count)
    # Evenly spaced indices across the whole video, including both ends.
    if n_samples == 1:
        indices = [frame_count // 2]
    else:
        step = (frame_count - 1) / (n_samples - 1)
        indices = [round(i * step) for i in range(n_samples)]

    results = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        timestamp = idx / fps
        frame_path = out_dir / f"frame_{idx:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        results.append((idx, timestamp, frame_path))
    cap.release()

    if not results:
        raise DetectorError("Frame extraction produced no usable frames.")
    return results


def analyze_video(path: Union[str, Path], workdir: Path) -> VideoAnalysisResult:
    """Full experimental pipeline: extract frames, run the real image
    detector on each, aggregate with a documented mean-score rule.

    Aggregation method (disclosed verbatim in the report): the mean of each
    frame's AI-class score and Real-class score across all sampled frames,
    then the same fixed decision thresholds as single-image analysis are
    applied to those means. This is a simple, transparent heuristic — not a
    validated video-level classifier — and the report says so explicitly.
    """
    frames = extract_evenly_spaced_frames(path, workdir / "frames")

    frame_results: List[FrameResult] = []
    for idx, ts, frame_path in frames:
        detection = predict_image(frame_path)
        frame_results.append(FrameResult(idx, ts, frame_path, detection))

    ai_scores = [
        fr.detection.scores.get("ai", fr.detection.scores.get("fake", 0.0))
        for fr in frame_results
    ]
    real_scores = [
        fr.detection.scores.get("real", fr.detection.scores.get("human", 0.0))
        for fr in frame_results
    ]
    mean_ai = sum(ai_scores) / len(ai_scores)
    mean_real = sum(real_scores) / len(real_scores)

    if mean_ai >= config.AI_SUSPECTED_THRESHOLD:
        category = ResultCategory.AI_GENERATED_SUSPECTED
    elif mean_real >= config.REAL_SUSPECTED_THRESHOLD:
        category = ResultCategory.REAL_SUSPECTED
    else:
        category = ResultCategory.INCONCLUSIVE

    validation = validate_video_file(path)

    return VideoAnalysisResult(
        frame_results=frame_results,
        aggregation_method=(
            "Mean of per-frame AI-class and Real-class scores across "
            f"{len(frame_results)} evenly spaced sampled frames, then the same "
            "fixed thresholds used for single-image analysis. This is a simple "
            "heuristic, not a validated temporal/video-level detector."
        ),
        mean_ai_score=mean_ai,
        mean_real_score=mean_real,
        category=category,
        frames_analyzed=len(frame_results),
        video_duration_seconds=validation.duration_seconds or 0.0,
    )
