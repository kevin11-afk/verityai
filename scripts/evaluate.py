#!/usr/bin/env python3
"""
Reproducible evaluation script for the detector and the RAG pipeline.

Honesty requirement from the project spec: if no real labeled evaluation
data is present, this script prints "Not evaluated" for every metric that
would need it, rather than inventing a number. It never fabricates results.

Detector evaluation expects a directory structured as:
    data/eval/images/real/*.jpg
    data/eval/images/ai/*.jpg
(both class folders; create this yourself from a licensed dataset — see
README "Evaluation instructions" for where to get one for free.)

RAG evaluation expects a JSONL file at data/eval/rag_eval.jsonl with lines:
    {"question": "...", "relevant_chunk_ids": ["doc.pdf:p3:0:abc123..."]}
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

EVAL_IMAGE_DIR = config.PROJECT_ROOT / "data" / "eval" / "images"
RAG_EVAL_FILE = config.PROJECT_ROOT / "data" / "eval" / "rag_eval.jsonl"


def evaluate_detector() -> None:
    print("=" * 70)
    print("DETECTOR EVALUATION")
    print("=" * 70)

    real_dir = EVAL_IMAGE_DIR / "real"
    ai_dir = EVAL_IMAGE_DIR / "ai"

    if not real_dir.is_dir() or not ai_dir.is_dir():
        print(f"Not evaluated: no labeled evaluation set found at {EVAL_IMAGE_DIR}")
        print("Expected subfolders 'real/' and 'ai/' with labeled images.")
        print("See README 'Evaluation instructions' for a free, licensed dataset to use.")
        return

    from src.detector.image_detector import predict_image
    from src.detector.base import ResultCategory
    import numpy as np
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score,
    )

    y_true, y_pred, y_score, latencies = [], [], [], []
    errors = []

    for label, directory in (("real", real_dir), ("ai", ai_dir)):
        for path in sorted(directory.glob("*")):
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                continue
            try:
                t0 = time.perf_counter()
                result = predict_image(path)
                latencies.append((time.perf_counter() - t0) * 1000)
            except Exception as exc:  # noqa: BLE001
                errors.append((str(path), str(exc)))
                continue

            y_true.append(1 if label == "ai" else 0)
            y_pred.append(1 if result.category == ResultCategory.AI_GENERATED_SUSPECTED else 0)
            y_score.append(result.scores.get("ai", result.scores.get("fake", 0.0)))

    n = len(y_true)
    if n == 0:
        print(f"Not evaluated: no readable images found under {EVAL_IMAGE_DIR}")
        return

    print(f"Evaluated {n} image(s) ({errors and len(errors)} error(s)).")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1:        {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Confusion matrix [[TN,FP],[FN,TP]]:\n{confusion_matrix(y_true, y_pred)}")
    if len(set(y_true)) == 2:
        print(f"ROC-AUC:   {roc_auc_score(y_true, y_score):.4f}")
    else:
        print("ROC-AUC:   Not evaluated (eval set has only one class present)")
    print(f"Mean inference latency: {np.mean(latencies):.1f}ms (n={len(latencies)})")

    if errors:
        print("\nFailure cases:")
        for path, msg in errors:
            print(f"  - {path}: {msg}")


def evaluate_rag() -> None:
    print()
    print("=" * 70)
    print("RAG EVALUATION")
    print("=" * 70)

    if not RAG_EVAL_FILE.exists():
        print(f"Not evaluated: no labeled question set found at {RAG_EVAL_FILE}")
        print('Create it as JSONL: {"question": "...", "relevant_chunk_ids": ["..."]}')
        return

    from src.rag.vector_store import VectorStore
    from src.rag.retrieval import HybridRetriever

    store = VectorStore()
    if store.count() == 0:
        print("Not evaluated: the vector store is empty. Run scripts/ingest_documents.py first.")
        return
    retriever = HybridRetriever(store)

    examples = [json.loads(line) for line in RAG_EVAL_FILE.read_text().splitlines() if line.strip()]
    if not examples:
        print(f"Not evaluated: {RAG_EVAL_FILE} contains no examples.")
        return

    recall_hits, mrr_sum, citation_coverage_hits, latencies = 0, 0.0, 0, []
    k = config.RETRIEVAL_TOP_K

    for ex in examples:
        question = ex["question"]
        relevant = set(ex.get("relevant_chunk_ids", []))
        if not relevant:
            continue

        t0 = time.perf_counter()
        passages = retriever.retrieve(question, top_k=k)
        latencies.append((time.perf_counter() - t0) * 1000)

        retrieved_ids = [p.chunk_id for p in passages]
        hit = any(cid in relevant for cid in retrieved_ids)
        recall_hits += int(hit)
        if passages:
            citation_coverage_hits += 1

        rank = next((i + 1 for i, cid in enumerate(retrieved_ids) if cid in relevant), None)
        if rank:
            mrr_sum += 1.0 / rank

    n = len(examples)
    print(f"Evaluated {n} question(s) at k={k}.")
    print(f"Recall@{k}:              {recall_hits / n:.4f}")
    print(f"Mean Reciprocal Rank:    {mrr_sum / n:.4f}")
    print(f"Citation coverage:       {citation_coverage_hits / n:.4f} "
          f"(fraction of questions that returned >=1 passage to cite)")
    print(f"Mean retrieval latency:  {sum(latencies) / len(latencies):.1f}ms")


if __name__ == "__main__":
    evaluate_detector()
    evaluate_rag()
