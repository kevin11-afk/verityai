#!/usr/bin/env python3
"""End-to-end checks against the REAL models and index (needs network the first
time for model downloads). Prints PASS/FAIL per check; exits 1 on any FAIL.

    python scripts/verify_pipeline.py [sample_image]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

fails = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global fails
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def main() -> None:
    from src.detector.image_detector import _load_pipeline, predict_image
    from src.rag.bootstrap import ensure_index_built
    from src.rag.generation import answer_research_question, generate_explanation
    from src.rag.retrieval import HybridRetriever
    from src.rag.vector_store import VectorStore

    # 1. detector labels + inference
    clf = _load_pipeline()
    id2label = dict(clf.model.config.id2label)
    check("detector id2label is {real, ai}", {str(v).lower() for v in id2label.values()} == {"real", "ai"}, str(id2label))
    if len(sys.argv) > 1:
        img = Path(sys.argv[1])
    else:
        import sklearn
        img = Path(sklearn.__file__).parent / "datasets" / "images" / "china.jpg"
    det = predict_image(img)
    check("scores sum to 1 (softmax)", abs(sum(det.scores.values()) - 1) < 1e-3, str(det.scores))
    check("score keys are real labels", set(det.scores) == {"real", "ai"})

    # 2. index + retrieval metadata
    store = VectorStore()
    ensure_index_built(store)
    n_kb_files = len([p for p in config.KNOWLEDGE_BASE_DIR.iterdir() if p.suffix in config.ALLOWED_DOC_EXTENSIONS and p.name.lower() != "readme.md"])
    allm = store.get_all_documents_with_metadata()
    sources = {m["source_file"] for m in allm["metadatas"]}
    check("every KB file is indexed", len(sources) == n_kb_files, f"{len(sources)}/{n_kb_files} files, {store.count()} chunks")
    check("all indexed source files exist on disk", all((config.KNOWLEDGE_BASE_DIR / s).exists() for s in sources))

    retriever = HybridRetriever(store)
    cases = {
        "Why do detectors fail on images from new generators?": "02_generalization_and_dataset_bias.md",
        "What is C2PA content credentials provenance?": "05_content_provenance_c2pa.md",
        "FaceForensics++ video deepfake benchmark": "03_video_and_deepfake_detection.md",
        "Which detector model was selected and why Apache license": "04_detector_model_cards.md",
        "precision recall ROC-AUC evaluation methodology": "06_evaluation_methodology.md",
    }
    corpus = {p.name: p.read_text() for p in config.KNOWLEDGE_BASE_DIR.glob("*.md")}
    for q, expected in cases.items():
        res = retriever.retrieve(q, top_k=5)
        check(f"relevant source in top-5: {expected[:2]}", any(r.source_file == expected for r in res),
              f"got {[r.source_file[:2] for r in res]}")
        check("  both retrievers contributed", any(r.vector_rank for r in res) and any(r.bm25_rank for r in res))
        # metadata correctness: passage text must really occur in the claimed file
        norm = lambda t: re.sub(r"\s+", " ", t)
        ok = all(norm(r.text)[:80] in norm(corpus.get(r.source_file, "")) for r in res)
        check("  passage text really comes from its claimed source_file", ok)

    # 3. grounding: citations/findings only from retrieved passages; no invented predictions
    res = retriever.retrieve("AI image detector limitations generalization", top_k=5)
    exp = generate_explanation(det, res, "x")
    check("one citation per retrieved passage, same order",
          [(c.title, c.source_file, c.page_number) for c in exp.citations] == [(r.title, r.source_file, r.page_number) for r in res])
    quotes = [re.search(r'"(.*)"$', f, re.S).group(1).rstrip("…") for f in exp.research_findings]
    check("every quoted finding is a prefix-substring of a retrieved passage",
          all(any(q.replace(" ", "")[:100] in r.text.replace("\n", "").replace(" ", "") for r in res) for q in quotes))
    check("detector summary reports exactly the model's category and scores",
          det.category.value in exp.detector_summary and all(f"{k.upper()}={v:.3f}" in exp.detector_summary for k, v in det.scores.items()))
    check("empty evidence -> explicit notice, no findings/citations",
          (lambda e: e.no_evidence_notice and not e.research_findings and not e.citations)(generate_explanation(det, [], "q")))
    check("empty evidence (assistant) -> explicit notice",
          bool(answer_research_question("q", []).no_evidence_notice))

    # 4. off-topic query should not be answered with citations
    off = retriever.retrieve("best recipe for chocolate chip cookies and pizza dough", top_k=5)
    check("off-topic query returns NO passages (insufficient-evidence path)", len(off) == 0, f"returned {len(off)} passages")

    print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILED'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
