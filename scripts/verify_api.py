#!/usr/bin/env python3
"""End-to-end API verification with the REAL detector, embeddings, ChromaDB and BM25
(no stubs). Uses an isolated temp directory for reports and uploads.

    python scripts/verify_api.py
"""
from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from api.main import create_app  # noqa: E402
from api.services.answer import clean_markdown  # noqa: E402
from api.settings import Settings  # noqa: E402
from src import config  # noqa: E402

fails = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global fails
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="verityai_api_"))
    settings = Settings(history_dir=tmp / "history", kb_upload_dir=tmp / "uploads", eval_dir=ROOT / "evaluation",
                        static_dir=ROOT / "product" / "static", cors_origins=(), persistence="ephemeral")
    client = TestClient(create_app(settings), raise_server_exceptions=False)

    h = client.get("/api/health").json()
    check("health ok", h["status"] in ("ok", "degraded") and h["thresholds"]["ai"] == config.AI_SUSPECTED_THRESHOLD)

    # --- image analysis with the real model ---------------------------------------------------
    import sklearn
    photo = Path(sklearn.__file__).parent / "datasets" / "images" / "china.jpg"
    r = client.post("/api/analyze/image", files={"file": ("china.jpg", photo.read_bytes(), "image/jpeg")})
    check("real image analysis returns 200", r.status_code == 200, r.text[:120] if r.status_code != 200 else "")
    rep = r.json()
    scores = rep["result"]["scores"]
    check("real labels are exactly {ai, real} and sum to 1", set(scores) == {"ai", "real"} and abs(sum(scores.values()) - 1) < 1e-3, str(scores))
    expected = ("AI-generated suspected" if scores["ai"] >= config.AI_SUSPECTED_THRESHOLD else
                "Real or non-synthetic suspected" if scores["real"] >= config.REAL_SUSPECTED_THRESHOLD else "Inconclusive")
    check("category follows the documented policy", rep["result"]["category"] == expected, rep["result"]["category"])
    check("report cites passages that exist on disk",
          all((config.KNOWLEDGE_BASE_DIR / p["source_file"]).exists() for p in rep["evidence"]["passages"]) and rep["evidence"]["passages"] != [])
    check("report is saved and downloadable", client.get(f"/api/reports/{rep['id']}/download?format=md").status_code == 200)
    check("report appears in history", client.get("/api/reports").json()["total"] == 1)

    # --- video ----------------------------------------------------------------------------------
    import cv2
    clip = tmp / "c.mp4"
    w = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 72))
    for i in range(30):
        w.write(np.full((72, 96, 3), (i * 8) % 255, dtype=np.uint8))
    w.release()
    v = client.post("/api/analyze/video", files={"file": ("c.mp4", clip.read_bytes(), "video/mp4")})
    check("real video analysis works and is experimental", v.status_code == 200 and "experimental" in v.json()["analysis_method"].lower(), v.text[:100] if v.status_code != 200 else "")

    # --- research -------------------------------------------------------------------------------
    for q, src in [("Why do detectors fail on images from new generators?", "02_generalization_and_dataset_bias.md"),
                   ("What are Content Credentials?", "05_content_provenance_c2pa.md"),
                   ("What does ROC-AUC measure?", "06_evaluation_methodology.md")]:
        a = client.post("/api/research/ask", json={"question": q}).json()
        by_id = {p["citation_id"]: p for p in a["passages"]}
        check(f"answers '{q[:38]}...' from the right document", not a["abstained"] and src in {p["source_file"] for p in a["passages"]})
        check("  every claim is verbatim from its cited passage",
              bool(a["answer"]) and all(c["text"] in clean_markdown(by_id[c["citation_id"]]["text"]) for c in a["answer"]))
    off = client.post("/api/research/ask", json={"question": "best recipe for chocolate chip cookies"}).json()
    check("off-topic question abstains with no passages", off["abstained"] and off["passages"] == [] and off["answer"] == [])

    # --- knowledge base ---------------------------------------------------------------------------
    doc = b"# Marmot Acoustics\n\nMarmots communicate with whistles that carry across alpine meadows for kilometres."
    up = client.post("/api/kb/documents", files={"file": ("marmots.md", doc, "text/markdown")}).json()
    check("document upload is indexed", up["status"] == "indexed")
    check("duplicate upload is detected", client.post("/api/kb/documents", files={"file": ("again.md", doc, "text/markdown")}).json()["status"] == "duplicate")
    found = client.post("/api/kb/search", json={"query": "marmots whistles alpine"}).json()
    check("uploaded document is retrievable", any(p["source_file"] == "marmots.md" for p in found["passages"]))
    check("removal needs confirmation", client.delete("/api/kb/documents/marmots.md").status_code == 400)
    check("confirmed removal works", client.delete("/api/kb/documents/marmots.md?confirm=marmots.md").status_code == 200)
    gone = client.post("/api/kb/search", json={"query": "marmots whistles alpine"}).json()
    check("removed document is no longer retrievable", all(p["source_file"] != "marmots.md" for p in gone["passages"]))

    # --- evaluation artifacts -----------------------------------------------------------------------
    e = client.get("/api/evaluation").json()
    check("evaluation artifacts are served", bool(e["detector"]) and bool(e["rag"]))
    if e["detector"]:
        prod = next(m for m in e["detector"]["models"] if m["is_production"])["variants"]["raw"]
        check("detector artifact has curves, CI and samples", len(prod["roc"]) > 10 and prod["auc_ci"][0] < prod["auc"] < prod["auc_ci"][1] and len(prod["samples"]["scores"]) == 200)

    # --- frontend --------------------------------------------------------------------------------------
    if (settings.static_dir / "index.html").exists():
        check("SPA is served with deep-link fallback", client.get("/product/").status_code == 200 and "VerityAI" in client.get("/product/reports/x").text)

    print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILED'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
