#!/usr/bin/env python3
"""Score a labeled image folder (<dir>/real, <dir>/ai) with one or more Hugging
Face image-classification detectors and print real, computed metrics.

    python scripts/compare_detectors.py data/eval_sets/raw_real_fake_images \
        Nahrawy/AIorNot umm-maybe/AI-image-detector

Scores are cached in <dir>/.scores_<model>.json so reruns are cheap. The AI
class is found from the model's own id2label (names containing ai/fake/
artificial/synthetic), never assumed. Softmax is applied explicitly.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

AI_HINTS = ("ai", "fake", "artificial", "synthetic", "generated", "sdxl")


def ai_label_index(id2label: dict) -> int:
    hits = [int(i) for i, n in id2label.items()
            if any(h in str(n).lower().replace("real", "") for h in AI_HINTS)]
    if len(hits) != 1:
        raise SystemExit(f"Cannot identify the AI class unambiguously from {id2label}")
    return hits[0]


def score_model(model_name: str, files: list[Path]) -> dict[str, float]:
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    proc = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name).eval()
    ai_idx = ai_label_index(model.config.id2label)
    print(f"  {model_name}: id2label={model.config.id2label} -> AI index {ai_idx}")
    out = {}
    for f in files:
        with Image.open(f) as im:
            inputs = proc(im.convert("RGB"), return_tensors="pt")
        with torch.no_grad():
            p = torch.softmax(model(**inputs).logits[0], -1)
        out[str(f)] = float(p[ai_idx])
    return out


def metrics(y, s, thr):
    from sklearn.metrics import roc_auc_score
    y, s = np.array(y), np.array(s)
    pred = s >= thr
    tp, fp, tn, fn = (pred & (y == 1)).sum(), (pred & (y == 0)).sum(), (~pred & (y == 0)).sum(), (~pred & (y == 1)).sum()
    return dict(auc=roc_auc_score(y, s), acc=(tp + tn) / len(y), tpr=tp / max(tp + fn, 1),
                tnr=tn / max(tn + fp, 1), fp=int(fp), fn=int(fn))


def main() -> None:
    root, models = Path(sys.argv[1]), sys.argv[2:]
    files, y = [], []
    for cls, lab in (("real", 0), ("ai", 1)):
        fs = sorted(p for p in (root / cls).glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
        files += fs
        y += [lab] * len(fs)
    print(f"{root}: {y.count(0)} real, {y.count(1)} ai")
    for m in models:
        cache = root / f".scores_{m.replace('/', '__')}.json"
        if cache.exists():
            scores = json.loads(cache.read_text())
        else:
            t0 = time.perf_counter()
            scores = score_model(m, files)
            cache.write_text(json.dumps(scores))
            print(f"  scored in {time.perf_counter() - t0:.0f}s")
        s = [scores[str(f)] for f in files]
        print(f"\n== {m}")
        for thr, name in ((0.5, "argmax (0.50)"), (0.65, "project policy (AI>=0.65)")):
            r = metrics(y, s, thr)
            print(f"  {name:28s} acc={r['acc']:.3f} AI-recall={r['tpr']:.3f} real-recall={r['tnr']:.3f} FP={r['fp']} FN={r['fn']}")
        print(f"  ROC-AUC={metrics(y, s, .5)['auc']:.3f}")


if __name__ == "__main__":
    main()
