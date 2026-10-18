"""Manual smoke test (pytest isn't installed in this sandbox yet — no network
to pip install it). Exercises every module that doesn't require torch/
transformers/chromadb/rank-bm25/sentence-transformers/streamlit at runtime,
using plain asserts. Run with: python3 _smoke_test.py
"""
import io
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

passed, failed = 0, 0


def check(name, fn):
    global passed, failed
    try:
        fn()
        print(f"OK   {name}")
        passed += 1
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL {name}: {exc}")
        traceback.print_exc()
        failed += 1


def test_config_imports():
    from src import config
    assert config.APP_NAME == "VerityAI"
    assert 0 < config.AI_SUSPECTED_THRESHOLD <= 1


def test_validation_valid_jpeg():
    from PIL import Image
    from src import validation
    buf = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buf, format="JPEG")
    result = validation.validate_image_bytes(buf.getvalue(), "x.jpg")
    assert result.ok, result.reason
    assert result.width == 64


def test_validation_rejects_fake_image():
    from src import validation
    result = validation.validate_image_bytes(b"not an image", "x.jpg")
    assert not result.ok


def test_validation_safe_temp_path_no_traversal():
    from src import validation, config
    p = validation.safe_temp_path(".jpg")
    assert p.parent == config.TMP_UPLOAD_DIR
    assert ".." not in str(p)


def test_validation_video_signature():
    from src import validation
    fake_mp4 = b"\x00\x00\x00\x18ftypisom" + b"0" * 50
    result = validation.validate_video_bytes(fake_mp4, "x.mp4")
    assert result.ok, result.reason
    bad = validation.validate_video_bytes(b"not a video" * 5, "x.mp4")
    assert not bad.ok


def test_detector_base_dataclass():
    from src.detector.base import DetectionResult, ResultCategory
    r = DetectionResult(
        scores={"ai": 0.9, "real": 0.1}, category=ResultCategory.AI_GENERATED_SUSPECTED,
        model_name="m", model_version_label="v", decision_policy="p",
        raw_top_label="ai", inference_latency_ms=1.0,
    )
    assert r.category.value == "AI-generated suspected"


def test_decide_category_logic():
    from src.detector.image_detector import _decide_category
    from src.detector.base import ResultCategory
    cat, policy = _decide_category({"ai": 0.9, "real": 0.1})
    assert cat == ResultCategory.AI_GENERATED_SUSPECTED
    cat, _ = _decide_category({"ai": 0.1, "real": 0.9})
    assert cat == ResultCategory.REAL_SUSPECTED
    cat, _ = _decide_category({"ai": 0.5, "real": 0.5})
    assert cat == ResultCategory.INCONCLUSIVE


def test_predict_image_mocked():
    from unittest.mock import patch
    from PIL import Image
    import tempfile, os
    from src.detector.base import ResultCategory

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "t.jpg"
        Image.new("RGB", (16, 16)).save(path)
        mock_output = [{"label": "ai", "score": 0.77}, {"label": "real", "score": 0.23}]
        with patch("src.detector.image_detector._load_pipeline") as mock_load:
            mock_load.return_value = lambda img: mock_output
            from src.detector.image_detector import predict_image
            result = predict_image(path)
        assert result.category == ResultCategory.AI_GENERATED_SUSPECTED
        assert result.scores["ai"] == 0.77


def test_chunking_preserves_metadata():
    from src.rag.chunking import chunk_document
    from src.rag.ingestion import LoadedDocument, LoadedPage
    doc = LoadedDocument(
        source_path=Path("paper.pdf"), title="Example", source_url="https://x.com",
        pages=[LoadedPage(page_number=1, text="Sentence. " * 200)],
    )
    chunks = chunk_document(doc, chunk_size=200, overlap=20)
    assert len(chunks) > 1
    assert all(c.title == "Example" and c.page_number == 1 for c in chunks)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_load_text_and_markdown():
    import tempfile
    from src.rag.ingestion import load_document
    with tempfile.TemporaryDirectory() as d:
        md_path = Path(d) / "doc.md"
        md_path.write_text("# My Title\n\nSome content here.")
        doc = load_document(md_path)
        assert doc.title == "My Title"
        assert "Some content" in doc.full_text


def test_reporting_round_trip():
    import json
    from src import reporting
    from src.detector.base import DetectionResult, ResultCategory
    from src.rag.generation import GeneratedExplanation, Citation

    detection = DetectionResult(
        scores={"ai": 0.9, "real": 0.1}, category=ResultCategory.AI_GENERATED_SUSPECTED,
        model_name="m", model_version_label="v", decision_policy="p",
        raw_top_label="ai", inference_latency_ms=1.0,
    )
    explanation = GeneratedExplanation(
        detector_summary="s", research_findings=["f"],
        citations=[Citation("T", "f.pdf", 1, None)], limitations=["l"], no_evidence_notice=None,
    )
    report = reporting.build_image_report(
        file_name="x.jpg", file_type="image/jpeg", file_size_bytes=10,
        image_width=10, image_height=10, detection=detection, explanation=explanation,
    )
    payload = json.loads(reporting.to_json(report))
    assert payload["predicted_category"] == "AI-generated suspected"
    md = reporting.to_markdown(report)
    assert "does not prove" in md.lower()


def test_generation_no_evidence_notice():
    from src.rag.generation import generate_explanation
    from src.detector.base import DetectionResult, ResultCategory
    detection = DetectionResult(
        scores={"ai": 0.9, "real": 0.1}, category=ResultCategory.AI_GENERATED_SUSPECTED,
        model_name="m", model_version_label="v", decision_policy="p",
        raw_top_label="ai", inference_latency_ms=1.0,
    )
    explanation = generate_explanation(detection, evidence=[], query_context="q")
    assert explanation.no_evidence_notice is not None
    assert explanation.research_findings == []


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        check(t.__name__, t)
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
