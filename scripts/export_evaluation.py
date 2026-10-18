#!/usr/bin/env python3
"""Export real evaluation artifacts for the Evaluation Lab (evaluation/*.json).

Everything is computed from measured scores, never typed in:

  detector  - reads cached per-image scores written by scripts/compare_detectors.py
              (data/eval_sets/<dataset>/.scores_<model>.json) and computes ROC, PR,
              calibration, bootstrap CIs and per-group breakdowns.
  rag       - runs the labeled questions in evaluation/rag_questions.jsonl through the
              real hybrid retriever (needs the embedding model) and computes Hit@k, MRR,
              abstention behaviour and a retrieval-mode ablation.

    python scripts/export_evaluation.py detector
    python scripts/export_evaluation.py rag
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

OUT = ROOT / "evaluation"
DATASET = "raw_real_fake_images"
MODELS = [
    # id, license, commercial use allowed, note
    ("Nahrawy/AIorNot", "Apache-2.0", True, "Production detector"),
    ("Organika/sdxl-detector", "CC-BY-NC-3.0", False, "Non-commercial license: cannot ship in a commercial product"),
    ("dima806/ai_vs_real_image_detection", "Apache-2.0", True, "Likely CIFAKE-trained"),
    ("umm-maybe/AI-image-detector", "CC-BY-4.0 (card also says ND)", False, "License terms inconsistent on the model card"),
]


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def git_commit() -> str:
    """Short commit hash of the code that produced the artifact, marked "+uncommitted" when the
    working tree has changes outside evaluation/ (so a stored hash is never over-claimed)."""
    try:
        head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain", "--", ".", ":!evaluation"], cwd=ROOT, text=True)
        return head + ("+uncommitted" if status.strip() else "")
    except Exception:  # noqa: BLE001
        return "unknown"


# --------------------------------------------------------------------------- detector
def _curve_points(xs, ys, ts, limit=120):
    idx = np.unique(np.linspace(0, len(xs) - 1, min(limit, len(xs))).astype(int))
    return [[float(xs[i]), float(ys[i]), float(ts[i]) if np.isfinite(ts[i]) else None] for i in idx]


def detector_block(y, s, groups, thr):
    from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve

    y, s = np.asarray(y), np.asarray(s)
    fpr, tpr, th = roc_curve(y, s)
    prec, rec, pth = precision_recall_curve(y, s)
    rng = np.random.default_rng(0)
    aucs = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            aucs.append(roc_auc_score(y[i], s[i]))
    bins = np.linspace(0, 1, 11)
    cal = []
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (s >= lo) & ((s < hi) | (hi == 1.0 and s <= hi))
        if m.sum():
            conf, acc = float(s[m].mean()), float(y[m].mean())
            ece += m.sum() / len(s) * abs(conf - acc)
            cal.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()), "mean_score": conf, "fraction_ai": acc})
    pred = s >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    tn, fn = int((~pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
    ai_sus, real_sus = s >= thr, (1 - s) >= thr
    inc = ~ai_sus & ~real_sus
    policy = {
        "ai_suspected": int(ai_sus.sum()), "real_suspected": int(real_sus.sum()), "inconclusive": int(inc.sum()),
        "ai_suspected_correct": int((y[ai_sus] == 1).sum()), "real_suspected_correct": int((y[real_sus] == 0).sum()),
    }
    grp = []
    for name, label, idxs in groups:
        idxs = np.asarray(idxs)
        k = int((s[idxs] >= thr).sum())
        grp.append({"name": name, "label": label, "n": int(len(idxs)), "flagged_ai": k,
                    "flagged_rate": k / len(idxs), "flagged_ci": wilson(k, len(idxs)),
                    "mean_ai_score": float(s[idxs].mean())})
    return {
        "auc": float(roc_auc_score(y, s)), "auc_ci": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
        "average_precision": float(average_precision_score(y, s)),
        "roc": _curve_points(fpr, tpr, th), "pr": _curve_points(rec, prec, np.append(pth, np.nan)),
        "calibration": {"bins": cal, "ece": float(ece), "brier": float(brier_score_loss(y, s)),
                        "valid": bool(len(y) >= 100 and sum(b["n"] >= 5 for b in cal) >= 5),
                        "note": "Reliability on a balanced 50/50 sample; real-world prevalence differs, so these "
                                "scores are not calibrated probabilities."},
        "at_policy_threshold": {
            "threshold": thr, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": (tp + tn) / len(y), "accuracy_ci": wilson(tp + tn, len(y)),
            "ai_recall": tp / max(tp + fn, 1), "ai_recall_ci": wilson(tp, tp + fn),
            "real_recall": tn / max(tn + fp, 1), "real_recall_ci": wilson(tn, tn + fp),
            "policy_outcomes": policy,
        },
        "groups": grp,
        "samples": {"labels": [int(v) for v in y], "scores": [round(float(v), 5) for v in s]},
    }


def export_detector():
    from PIL import Image

    base = ROOT / "data" / "eval_sets"
    variants = {"raw": base / DATASET, "normalized": base / f"{DATASET}_normalized"}
    if not variants["raw"].exists():
        sys.exit(f"Missing {variants['raw']}. Run scripts/fetch_eval_data.py and scripts/compare_detectors.py first.")

    files, y = [], []
    for cls, lab in (("real", 0), ("ai", 1)):
        fs = sorted(p for p in (variants["raw"] / cls).glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
        files += fs
        y += [lab] * len(fs)
    sizes = []
    for f in files:
        with Image.open(f) as im:
            sizes.append(im.size)
    gmap: dict = {}
    for i, (lab, sz) in enumerate(zip(y, sizes)):
        gmap.setdefault(("AI" if lab else "Real", sz), []).append(i)
    groups = []
    for (kind, sz), idxs in sorted(gmap.items(), key=lambda kv: (kv[0][0], -len(kv[1]))):
        if len(idxs) >= 5:
            groups.append((f"{kind} · {sz[0]}×{sz[1]} px", 1 if kind == "AI" else 0, idxs))
    thr = config.AI_SUSPECTED_THRESHOLD

    models = []
    for mid, lic, commercial, note in MODELS:
        entry = {"id": mid, "license": lic, "commercial_use": commercial, "note": note,
                 "is_production": mid == config.DETECTOR_MODEL_NAME, "variants": {}}
        for vname, vdir in variants.items():
            cache = vdir / f".scores_{mid.replace('/', '__')}.json"
            if not cache.exists():
                continue
            raw = json.loads(cache.read_text())
            # Cache keys are paths as written by compare_detectors.py (relative or absolute);
            # match on (class folder, file stem), which is identical across both variants.
            lookup = {(Path(k).parent.name, Path(k).stem): v for k, v in raw.items()}
            s = [lookup[(f.parent.name, f.stem)] for f in files]
            entry["variants"][vname] = detector_block(y, s, groups, thr)
        if entry["variants"]:
            models.append(entry)

    artifact = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "policy": {"ai_threshold": config.AI_SUSPECTED_THRESHOLD, "real_threshold": config.REAL_SUSPECTED_THRESHOLD,
                   "production_model": config.DETECTOR_MODEL_NAME},
        "dataset": {
            "name": "date3k2/raw_real_fake_images", "split": "test",
            "sampling": "100 real + 100 AI images, evenly spaced across the split",
            "n_real": y.count(0), "n_ai": y.count(1),
            "license_note": "The dataset card states no license. Only scores and statistics are published here; "
                            "no images are redistributed.",
            "variants": {"raw": "Original files", "normalized":
                         "Center-cropped square, resized to 512 px, JPEG q90 (controls for size/format shortcuts)"},
            "limitations": [
                "200 images from one dataset; confidence intervals are wide.",
                "Real photos are mostly ~800 px; AI images are square 512/1024/1536 px (see group breakdown).",
                "Generators in this set are older than the newest commercial models; a miss on a modern generator "
                "(e.g. GPT-image) has been observed in manual use and is not captured by these numbers.",
            ],
        },
        "models": models,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "detector_eval.json").write_text(json.dumps(artifact, indent=1))
    prod = next(m for m in models if m["is_production"])["variants"]["raw"]
    print(f"detector_eval.json: {len(models)} models, production AUC {prod['auc']:.3f} "
          f"(95% CI {prod['auc_ci'][0]:.3f}-{prod['auc_ci'][1]:.3f}), n={len(y)}")


# --------------------------------------------------------------------------- RAG
def export_rag():
    from api.services.answer import clean_markdown, compose_answer  # noqa: F401
    from src.rag.bootstrap import ensure_index_built
    from src.rag.retrieval import HybridRetriever, _tokenize
    from src.rag.vector_store import VectorStore

    questions = [json.loads(l) for l in (OUT / "rag_questions.jsonl").read_text().splitlines() if l.strip()]
    # Isolated index built only from the built-in knowledge_base/ documents, so results never depend on
    # whatever uploads happen to be in a developer's local chroma_db.
    import tempfile
    store = VectorStore(persist_directory=tempfile.mkdtemp(prefix="verityai_eval_"), collection_name="rag_eval")
    ensure_index_built(store)
    retriever = HybridRetriever(store)
    meta = store.get_all_documents_with_metadata()
    source_of = {cid: m["source_file"] for cid, m in zip(meta["ids"], meta["metadatas"])}

    def first_rank(sources, expected):
        for i, s in enumerate(sources, start=1):
            if s in expected:
                return i
        return None

    rows = []
    for q in questions:
        k = config.RETRIEVAL_TOP_K
        passages, diag = retriever.retrieve_with_diagnostics(q["question"], top_k=k)
        sources = [p.source_file for p in passages]
        # source-level ranking, de-duplicated, preserving order
        dedup = list(dict.fromkeys(sources))
        # mode ablation (ignores the relevance gate so the rankers can be compared fairly)
        vec = store.query(q["question"], top_k=max(k * 3, 10))
        vec_src = list(dict.fromkeys(source_of[i] for i in vec["ids"][0]))
        bm = retriever._bm25_ranked_ids(q["question"], max(k * 3, 10))
        bm_src = list(dict.fromkeys(source_of[i] for i in bm))
        # one hybrid ranking without the gate, for the ablation row
        gate = config.MAX_BEST_VECTOR_DISTANCE
        config.MAX_BEST_VECTOR_DISTANCE = 9.0
        ungated = [p.source_file for p in retriever.retrieve(q["question"], top_k=k)]
        config.MAX_BEST_VECTOR_DISTANCE = gate
        ungated = list(dict.fromkeys(ungated))
        claims = compose_answer(q["question"], [type("P", (), {"text": p.text, "citation_id": i})
                                                for i, p in enumerate(passages, start=1)]) if passages else []
        grounded = all(c[0] in clean_markdown(passages[c[1] - 1].text) for c in claims)
        exp = set(q["expected_sources"])
        rows.append({
            "id": q["id"], "category": q["category"], "question": q["question"], "expected_sources": q["expected_sources"],
            "retrieved_sources": dedup, "abstained": diag.abstained, "best_vector_distance": diag.best_vector_distance,
            "first_rank": first_rank(dedup, exp) if exp else None,
            "rank_vector": first_rank(vec_src, exp) if exp else None, "rank_bm25": first_rank(bm_src, exp) if exp else None,
            "rank_hybrid": first_rank(ungated, exp) if exp else None,
            "n_claims": len(claims), "claims_grounded": grounded,
        })

    ans = [r for r in rows if r["category"] == "answerable"]
    refuse = [r for r in rows if r["category"] != "answerable"]

    def hit(rank_key, kk):
        return sum(1 for r in ans if r[rank_key] is not None and r[rank_key] <= kk)

    def mrr(rank_key):
        return float(np.mean([1 / r[rank_key] if r[rank_key] else 0 for r in ans]))

    rng = np.random.default_rng(0)
    rr = np.array([1 / r["first_rank"] if r["first_rank"] else 0 for r in ans])
    boots = [float(rr[rng.integers(0, len(rr), len(rr))].mean()) for _ in range(2000)]
    # abstention sweep over the distance gate: uses only measured best distances
    sweep = []
    for t in np.round(np.arange(0.40, 1.001, 0.05), 2):
        ans_answered = sum(1 for r in ans if r["best_vector_distance"] is not None and r["best_vector_distance"] <= t)
        ref_abstained = sum(1 for r in refuse if r["best_vector_distance"] is None or r["best_vector_distance"] > t)
        sweep.append({"threshold": float(t), "answerable_answered": ans_answered, "answerable_total": len(ans),
                      "refusals_correct": ref_abstained, "refusals_total": len(refuse)})
    by_cat = {}
    for cat in ("unanswerable_in_domain", "off_topic"):
        rs = [r for r in rows if r["category"] == cat]
        k = sum(r["abstained"] for r in rs)
        by_cat[cat] = {"n": len(rs), "abstained": k, "rate": k / len(rs), "ci": wilson(k, len(rs))}
    n = len(ans)
    artifact = {
        "schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit(),
        "config": {"top_k": config.RETRIEVAL_TOP_K, "distance_threshold": config.MAX_BEST_VECTOR_DISTANCE,
                   "embedding_model": config.EMBEDDING_MODEL_NAME, "chunks": store.count(),
                   "documents": len(set(source_of.values()))},
        "question_set": {"answerable": n, "unanswerable_in_domain": by_cat["unanswerable_in_domain"]["n"],
                         "off_topic": by_cat["off_topic"]["n"],
                         "labeling": "Hand-written questions with expected source documents (document-level labels). "
                                     "Authored by the developer/assistant who also read the documents, so wording "
                                     "overlap may flatter retrieval; passage-level labels were not created."},
        "retrieval": {
            "hit_at_1": {"k": hit("first_rank", 1), "n": n, "rate": hit("first_rank", 1) / n, "ci": wilson(hit("first_rank", 1), n)},
            "hit_at_3": {"k": hit("first_rank", 3), "n": n, "rate": hit("first_rank", 3) / n, "ci": wilson(hit("first_rank", 3), n)},
            "hit_at_5": {"k": hit("first_rank", 5), "n": n, "rate": hit("first_rank", 5) / n, "ci": wilson(hit("first_rank", 5), n)},
            "mrr": float(rr.mean()), "mrr_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "citation_coverage": {"k": sum(1 for r in ans if not r["abstained"]), "n": n},
            "wrongly_abstained": [r["id"] for r in ans if r["abstained"]],
        },
        "ablation": [
            {"mode": "BM25 only", "hit_at_1": hit("rank_bm25", 1) / n, "hit_at_5": hit("rank_bm25", 5) / n, "mrr": mrr("rank_bm25")},
            {"mode": "Vector only", "hit_at_1": hit("rank_vector", 1) / n, "hit_at_5": hit("rank_vector", 5) / n, "mrr": mrr("rank_vector")},
            {"mode": "Hybrid (RRF)", "hit_at_1": hit("rank_hybrid", 1) / n, "hit_at_5": hit("rank_hybrid", 5) / n, "mrr": mrr("rank_hybrid")},
        ],
        "abstention": by_cat,
        "abstention_sweep": sweep,
        "groundedness": {"claims_total": sum(r["n_claims"] for r in rows),
                         "answers_fully_grounded": sum(1 for r in rows if r["claims_grounded"]), "answers": len(rows),
                         "method": "Every answer claim must be a verbatim substring of its cited passage."},
        "rows": rows,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "rag_eval.json").write_text(json.dumps(artifact, indent=1))
    r = artifact["retrieval"]
    print(f"rag_eval.json: hit@1 {r['hit_at_1']['rate']:.2f}  hit@5 {r['hit_at_5']['rate']:.2f}  MRR {r['mrr']:.3f}  "
          f"off-topic abstained {by_cat['off_topic']['abstained']}/{by_cat['off_topic']['n']}  "
          f"unanswerable-in-domain abstained {by_cat['unanswerable_in_domain']['abstained']}/{by_cat['unanswerable_in_domain']['n']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["detector", "rag", "all"])
    a = ap.parse_args()
    if a.what in ("detector", "all"):
        export_detector()
    if a.what in ("rag", "all"):
        export_rag()
