"""
Tests for the detector module.

Per the project's explicit requirement, mocked model outputs are used ONLY
here in unit tests, never in the production app. These tests verify the
decision-policy logic and error handling around a (mocked) model, not the
real model's accuracy.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src import config
from src.detector.base import DetectorError, ResultCategory
from src.detector.image_detector import _decide_category


class TestDecisionPolicy:
    def test_high_ai_score_is_ai_suspected(self):
        category, policy = _decide_category({"ai": 0.90, "real": 0.10})
        assert category == ResultCategory.AI_GENERATED_SUSPECTED
        assert "0.90" not in policy  # policy text states thresholds, not this call's scores

    def test_high_real_score_is_real_suspected(self):
        category, _ = _decide_category({"ai": 0.10, "real": 0.90})
        assert category == ResultCategory.REAL_SUSPECTED

    def test_mid_scores_are_inconclusive(self):
        category, _ = _decide_category({"ai": 0.50, "real": 0.50})
        assert category == ResultCategory.INCONCLUSIVE

    def test_boundary_at_exact_threshold_counts_as_suspected(self):
        category, _ = _decide_category({"ai": config.AI_SUSPECTED_THRESHOLD, "real": 0.0})
        assert category == ResultCategory.AI_GENERATED_SUSPECTED

    def test_alternate_label_casing_is_handled(self):
        """Some model cards use 'FAKE'/'REAL' instead of 'ai'/'real'."""
        category, _ = _decide_category({"fake": 0.95, "real": 0.05})
        assert category == ResultCategory.AI_GENERATED_SUSPECTED

    def test_missing_labels_defaults_to_zero_not_crash(self):
        category, _ = _decide_category({})
        assert category == ResultCategory.INCONCLUSIVE


class TestPredictImageMockedPipeline:
    """Mocks only the HF pipeline call itself (never the decision logic),
    so we can test predict_image()'s plumbing without downloading a real
    model or hitting the network in CI."""

    def test_predict_image_builds_correct_result_shape(self, tmp_path):
        from PIL import Image
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (32, 32), color=(1, 2, 3)).save(img_path)

        mock_output = [{"label": "ai", "score": 0.82}, {"label": "real", "score": 0.18}]

        with patch("src.detector.image_detector._load_pipeline") as mock_load:
            mock_load.return_value = lambda img: mock_output
            from src.detector.image_detector import predict_image
            result = predict_image(img_path)

        assert result.raw_top_label == "ai"
        assert result.category == ResultCategory.AI_GENERATED_SUSPECTED
        assert result.scores == {"ai": 0.82, "real": 0.18}
        assert result.error is None
        assert result.inference_latency_ms >= 0

    def test_predict_image_raises_detector_error_on_pipeline_failure(self, tmp_path):
        from PIL import Image
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (32, 32)).save(img_path)

        def _raise(_img):
            raise RuntimeError("simulated model failure")

        with patch("src.detector.image_detector._load_pipeline") as mock_load:
            mock_load.return_value = _raise
            from src.detector.image_detector import predict_image
            with pytest.raises(DetectorError):
                predict_image(img_path)

    def test_predict_image_raises_on_corrupt_file(self, tmp_path):
        bad_path = tmp_path / "corrupt.jpg"
        bad_path.write_bytes(b"not an image")

        with patch("src.detector.image_detector._load_pipeline") as mock_load:
            mock_load.return_value = lambda img: [{"label": "ai", "score": 1.0}]
            from src.detector.image_detector import predict_image
            with pytest.raises(DetectorError):
                predict_image(bad_path)
