"""API tests. The detector and embedding model are stubbed for speed; everything
else (validation, retrieval, ChromaDB, BM25, history, KB management) is real.
scripts/verify_api.py runs the same endpoints against the real models."""
from __future__ import annotations

import io
import re
import sys
import zlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("chromadb")
pytest.importorskip("rank_bm25")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from api.main import create_app  # noqa: E402
from api.services.answer import clean_markdown  # noqa: E402
from api.settings import Settings  # noqa: E402
from src import config  # noqa: E402
from src.detector.base import DetectionResult, DetectorError, ResultCategory  # noqa: E402
from src.rag.vector_store import VectorStore  # noqa: E402

_WORD = re.compile(r"[a-z0-9]+")


def _bow_embed(texts):
    """Deterministic bag-of-words hashing embedding: shared words => small cosine distance."""
    out = np.zeros((len(texts), 8192), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in _WORD.findall(t.lower()):
            out[i, zlib.crc32(w.encode()) % 8192] += 1.0
        n = np.linalg.norm(out[i])
        if n:
            out[i] /= n
    return out


def _detection(ai=0.8):
    return DetectionResult(
        scores={"ai": ai, "real": 1 - ai},
        category=ResultCategory.AI_GENERATED_SUSPECTED if ai >= 0.65 else ResultCategory.INCONCLUSIVE,
        model_name="stub/model", model_version_label="stub model v0", decision_policy="stub policy",
        raw_top_label="ai", inference_latency_ms=12.0,
    )


def _png(size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (30, 90, 160)).save(buf, format="PNG")
    return buf.getvalue()


def _mp4(path: Path) -> bytes:
    import cv2
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(30):
        w.write(np.full((48, 64, 3), i * 8 % 255, dtype=np.uint8))
    w.release()
    return path.read_bytes()


@pytest.fixture
def env(tmp_path, monkeypatch):
    settings = Settings(
        history_dir=tmp_path / "history", kb_upload_dir=tmp_path / "uploads", eval_dir=tmp_path / "eval",
        static_dir=tmp_path / "static", cors_origins=(), persistence="ephemeral",
    )
    monkeypatch.setattr(config, "MAX_BEST_VECTOR_DISTANCE", 0.9)
    store_dir = tmp_path / "chroma"
    with patch("src.rag.vector_store.embed_texts", side_effect=_bow_embed), \
         patch("api.services.knowledge.VectorStore", lambda: VectorStore(persist_directory=store_dir,
                                                                        collection_name="api_test")), \
         patch("api.services.analysis.predict_image", side_effect=lambda p: _detection()):
        yield settings, TestClient(create_app(settings), raise_server_exceptions=False), tmp_path


def test_health_reports_real_configuration(env):
    _, client, _ = env
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["thresholds"] == {"ai": config.AI_SUSPECTED_THRESHOLD, "real": config.REAL_SUSPECTED_THRESHOLD}
    assert body["limits"]["max_image_mb"] == config.MAX_IMAGE_SIZE_MB
    assert body["persistence_mode"] == "ephemeral" and "erased" in body["persistence_note"]
    assert body["detector_version_label"] == config.DETECTOR_VERSION_LABEL


def test_image_analysis_end_to_end_and_report_lifecycle(env):
    _, client, _ = env
    r = client.post("/api/analyze/image", files={"file": ("sample.png", _png(), "image/png")})
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["kind"] == "image" and rep["file"]["name"] == "sample.png"
    assert rep["file"]["width"] == 64 and len(rep["file"]["sha256"]) == 64
    assert rep["result"]["scores"] == {"ai": 0.8, "real": pytest.approx(0.2)}
    assert rep["result"]["category"] == "AI-generated suspected"
    assert rep["evidence"]["passages"], "seed knowledge base should yield evidence"
    ids = [p["citation_id"] for p in rep["evidence"]["passages"]]
    assert ids == list(range(1, len(ids) + 1))
    # report is persisted and retrievable
    assert client.get(f"/api/reports/{rep['id']}").json()["id"] == rep["id"]
    listing = client.get("/api/reports").json()
    assert listing["total"] == 1 and listing["reports"][0]["file_name"] == "sample.png"
    assert client.get("/api/reports", params={"q": "nomatch"}).json()["total"] == 0
    js = client.get(f"/api/reports/{rep['id']}/download", params={"format": "json"})
    assert js.status_code == 200 and "attachment" in js.headers["content-disposition"]
    assert js.json()["id"] == rep["id"]
    md = client.get(f"/api/reports/{rep['id']}/download", params={"format": "md"})
    assert "Model prediction" in md.text and "does not prove" in md.text.lower()
    assert client.get(f"/api/reports/{rep['id']}/download", params={"format": "pdf"}).status_code == 400
    assert client.delete(f"/api/reports/{rep['id']}").status_code == 200
    assert client.get(f"/api/reports/{rep['id']}").status_code == 404


def test_report_ids_cannot_traverse_paths(env):
    _, client, _ = env
    assert client.get("/api/reports/..%2F..%2Fetc%2Fpasswd").status_code == 404
    assert client.get("/api/reports/not-a-real-id").status_code == 404


def test_invalid_uploads_are_rejected_with_clear_errors(env, monkeypatch):
    _, client, _ = env
    r = client.post("/api/analyze/image", files={"file": ("fake.jpg", b"definitely not an image", "image/jpeg")})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_file"
    r = client.post("/api/analyze/image", files={"file": ("x.exe", _png(), "application/octet-stream")})
    assert r.status_code == 422 and "extension" in r.json()["error"]["message"].lower()
    monkeypatch.setattr(config, "MAX_IMAGE_SIZE_MB", 0)
    r = client.post("/api/analyze/image", files={"file": ("big.png", _png(), "image/png")})
    assert r.status_code == 413 and r.json()["error"]["code"] == "file_too_large"
    assert client.post("/api/analyze/image").status_code == 422


def test_detector_failure_is_a_503_not_a_fabricated_result(env):
    _, client, _ = env
    with patch("api.services.analysis.predict_image", side_effect=DetectorError("weights missing")):
        r = client.post("/api/analyze/image", files={"file": ("a.png", _png(), "image/png")})
    assert r.status_code == 503 and r.json()["error"]["code"] == "detector_unavailable"
    assert client.get("/api/reports").json()["total"] == 0


def test_unexpected_errors_do_not_leak_internals(env):
    _, client, _ = env
    with patch("api.services.analysis.predict_image", side_effect=RuntimeError("secret /path/leak")):
        r = client.post("/api/analyze/image", files={"file": ("a.png", _png(), "image/png")})
    assert r.status_code == 500 and "leak" not in r.text and r.json()["error"]["code"] == "internal_error"


def test_video_analysis_returns_frames_and_is_labelled_experimental(env, tmp_path):
    _, client, tmp = env
    data = _mp4(tmp / "c.mp4")
    r = client.post("/api/analyze/video", files={"file": ("c.mp4", data, "video/mp4")})
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["kind"] == "video" and rep["video"]["frames_analyzed"] >= 2
    assert len(rep["video"]["frames"]) == rep["video"]["frames_analyzed"]
    assert "experimental" in rep["analysis_method"].lower()
    assert any("frame-level" in x for x in rep["limitations"])
    bad = client.post("/api/analyze/video", files={"file": ("c.mp4", b"x" * 64, "video/mp4")})
    assert bad.status_code == 422


def test_research_answers_are_verbatim_from_cited_passages(env):
    _, client, _ = env
    r = client.post("/api/research/ask", json={"question": "Why do detectors fail on new generators?"})
    assert r.status_code == 200
    body = r.json()
    assert not body["abstained"] and body["passages"]
    by_id = {p["citation_id"]: p for p in body["passages"]}
    assert body["answer"], "expected at least one extractive claim"
    for claim in body["answer"]:
        assert claim["citation_id"] in by_id
        assert claim["text"] in clean_markdown(by_id[claim["citation_id"]]["text"])
    for p in body["passages"]:  # citations point at real documents
        assert (config.KNOWLEDGE_BASE_DIR / p["source_file"]).exists()
    assert body["diagnostics"]["best_vector_distance"] <= config.MAX_BEST_VECTOR_DISTANCE
    # The diagnostics contract the frontend depends on must survive serialization.
    assert body["diagnostics"]["gate_passed_by"] in ("vector", "lexical")
    assert "lexical_coverage" in body["diagnostics"]


def test_research_abstains_on_unrelated_questions(env):
    _, client, _ = env
    r = client.post("/api/research/ask", json={"question": "zzqx plumbing recipe xylophone qwerty"})
    body = r.json()
    assert body["abstained"] and body["passages"] == [] and body["answer"] == []
    assert "does not contain" in body["notice"]
    assert body["diagnostics"]["abstained"] and body["diagnostics"]["abstain_reason"]
    assert body["diagnostics"]["gate_passed_by"] is None


def test_research_input_validation(env):
    _, client, _ = env
    assert client.post("/api/research/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/api/research/ask", json={"question": "x" * 600}).status_code == 422
    assert client.post("/api/research/ask", json={"question": "valid question", "top_k": 99}).status_code == 422


def test_knowledge_base_management(env):
    settings, client, _ = env
    docs = client.get("/api/kb/documents").json()
    seeds = [d for d in docs["documents"] if d["origin"] == "seed"]
    assert len(seeds) >= 6 and all(not d["removable"] for d in seeds)
    assert "readme.md" not in {d["source_file"].lower() for d in docs["documents"]}

    text = b"# Marmot Acoustics\n\nMarmots communicate with whistles that carry across alpine meadows."
    up = client.post("/api/kb/documents", files={"file": ("marmots.md", text, "text/markdown")},
                     data={"source_url": "https://example.org/marmots"})
    assert up.status_code == 200 and up.json()["status"] == "indexed" and up.json()["chunks_added"] >= 1
    assert up.json()["document"]["removable"] and up.json()["document"]["source_url"] == "https://example.org/marmots"

    dup = client.post("/api/kb/documents", files={"file": ("copy.md", text, "text/markdown")}).json()
    assert dup["status"] == "duplicate" and dup["chunks_added"] == 0 and "marmots.md" in dup["message"]

    clash = client.post("/api/kb/documents", files={"file": ("marmots.md", text + b" more", "text/markdown")})
    assert clash.status_code == 409

    found = client.post("/api/kb/search", json={"query": "marmots whistles alpine meadows"}).json()
    assert any(p["source_file"] == "marmots.md" for p in found["passages"])

    assert client.delete("/api/kb/documents/marmots.md").status_code == 400  # needs confirmation
    assert client.delete("/api/kb/documents/marmots.md", params={"confirm": "wrong.md"}).status_code == 400
    assert client.delete("/api/kb/documents/marmots.md", params={"confirm": "marmots.md"}).status_code == 200
    assert not (settings.kb_upload_dir / "marmots.md").exists()
    after = client.post("/api/kb/search", json={"query": "marmots whistles alpine meadows"}).json()
    assert all(p["source_file"] != "marmots.md" for p in after["passages"])

    seed_name = seeds[0]["source_file"]
    r = client.delete(f"/api/kb/documents/{seed_name}", params={"confirm": seed_name})
    assert r.status_code == 403 and r.json()["error"]["code"] == "read_only"
    assert (config.KNOWLEDGE_BASE_DIR / seed_name).exists()

    assert client.post("/api/kb/reindex").status_code == 400
    ok = client.post("/api/kb/reindex", params={"confirm": "true"})
    assert ok.status_code == 200 and ok.json()["chunks"] > 0


def test_kb_upload_validation_and_filename_sanitising(env):
    settings, client, _ = env
    assert client.post("/api/kb/documents", files={"file": ("a.exe", b"MZ", "x/y")}).status_code == 422
    assert client.post("/api/kb/documents", files={"file": ("a.pdf", b"not a pdf", "application/pdf")}).status_code == 422
    assert client.post("/api/kb/documents", files={"file": ("a.txt", b"", "text/plain")}).status_code == 422
    assert client.post("/api/kb/documents", files={"file": ("a.txt", b"bin\x00ary", "text/plain")}).status_code == 422
    bad_url = client.post("/api/kb/documents", files={"file": ("a.txt", b"hello world text here", "text/plain")},
                          data={"source_url": "javascript:alert(1)"})
    assert bad_url.status_code == 422
    r = client.post("/api/kb/documents",
                    files={"file": ("../../evil name.md", b"# Title\n\nSome harmless text for indexing here.", "text/markdown")})
    assert r.status_code == 200
    name = r.json()["document"]["source_file"]
    assert "/" not in name and ".." not in name
    assert (settings.kb_upload_dir / name).is_file()
    assert not (settings.kb_upload_dir.parent / "evil name.md").exists()


def test_evaluation_endpoint_reports_not_evaluated_without_artifacts(env):
    settings, client, _ = env
    assert client.get("/api/evaluation").json() == {"detector": None, "rag": None}
    settings.eval_dir.mkdir()
    (settings.eval_dir / "detector_eval.json").write_text('{"dataset": {"name": "x"}}')
    assert client.get("/api/evaluation").json()["detector"]["dataset"]["name"] == "x"


def test_spa_fallback_serves_index_for_client_routes(env):
    settings, client, tmp = env
    (settings.static_dir / "assets").mkdir(parents=True)
    (settings.static_dir / "index.html").write_text("<html>SPA</html>")
    (settings.static_dir / "assets" / "a.js").write_text("console.log(1)")
    client = TestClient(create_app(settings), raise_server_exceptions=False)
    assert client.get("/product/").text == "<html>SPA</html>"
    assert client.get("/product/research").text == "<html>SPA</html>"  # refresh on a deep link
    js = client.get("/product/assets/a.js")
    assert js.status_code == 200 and "immutable" in js.headers["cache-control"]
    assert client.get("/", follow_redirects=False).headers["location"] == "/product/"


def test_cors_only_for_configured_origins(env, tmp_path):
    settings, _, _ = env
    from dataclasses import replace
    client = TestClient(create_app(replace(settings, cors_origins=("https://ui.example",))))
    ok = client.get("/api/health", headers={"Origin": "https://ui.example"})
    assert ok.headers.get("access-control-allow-origin") == "https://ui.example"
    other = client.get("/api/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in other.headers
