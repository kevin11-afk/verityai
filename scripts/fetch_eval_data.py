#!/usr/bin/env python3
"""Download a small, balanced, evenly-spaced sample of a public labeled
real-vs-AI image dataset from the Hugging Face datasets-server (free, no
account) into data/eval_sets/<name>/{real,ai}/.

Evaluation data stays local and untracked (see .gitignore). Check each
dataset's own license/terms before reuse; this script only fetches a sample.

    python scripts/fetch_eval_data.py date3k2/raw_real_fake_images --split test --per-class 100
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

ROWS_API = "https://datasets-server.huggingface.co/rows"
INFO_API = "https://datasets-server.huggingface.co/info"
CHUNK = 10  # rows per request; samples are spread across the whole split


def _class_dir_name(label_name: str) -> str:
    n = label_name.lower()
    return "ai" if any(k in n for k in ("fake", "ai", "synthetic", "generated")) and "real" not in n else "real"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--split", default="test")
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--out", default=None, help="default: data/eval_sets/<dataset name>")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    out = Path(args.out) if args.out else root / "data" / "eval_sets" / args.dataset.split("/")[-1]

    info = requests.get(INFO_API, params={"dataset": args.dataset}, timeout=60).json()["dataset_info"]["default"]
    names = info["features"]["label"]["names"]
    total = info["splits"][args.split]["num_examples"]
    print(f"{args.dataset}[{args.split}]: {total} rows, labels={names}")

    counts = {"real": 0, "ai": 0}
    for d in counts:
        (out / d).mkdir(parents=True, exist_ok=True)
        counts[d] = len(list((out / d).glob("*")))

    n_chunks = max(1, args.per_class * 2 // CHUNK * 3)  # oversample positions, stop when full
    stride = max(CHUNK, total // n_chunks)
    for pos in range(0, total, stride):
        if min(counts.values()) >= args.per_class:
            break
        r = requests.get(ROWS_API, params={"dataset": args.dataset, "config": "default",
                                           "split": args.split, "offset": pos, "length": CHUNK}, timeout=120)
        if r.status_code != 200:
            print(f"  offset {pos}: HTTP {r.status_code}, skipping", file=sys.stderr)
            continue
        for row in r.json()["rows"]:
            cls = _class_dir_name(names[row["row"]["label"]])
            if counts[cls] >= args.per_class:
                continue
            img = requests.get(row["row"]["image"]["src"], timeout=120)
            if img.status_code != 200:
                continue
            ext = ".png" if img.headers.get("content-type", "").endswith("png") else ".jpg"
            (out / cls / f"{args.split}_{row['row_idx']:07d}{ext}").write_bytes(img.content)
            counts[cls] += 1
        print(f"  offset {pos}: {counts}")
    print(f"Done: {counts} -> {out}")


if __name__ == "__main__":
    main()
