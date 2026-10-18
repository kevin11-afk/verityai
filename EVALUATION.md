# Detector evaluation (2026-10-18)

First measurement of detector accuracy in this project. **Small sample,
single dataset: read the limits below before quoting any number.**

## Setup
- Data: 100 real + 100 AI images sampled evenly across the `test` split of
  [`date3k2/raw_real_fake_images`](https://huggingface.co/datasets/date3k2/raw_real_fake_images)
  (Hugging Face; the dataset card states no license, so the sample is kept
  local and git-ignored). Fetched with `scripts/fetch_eval_data.py`.
- Real images are mostly ~800 px photos; AI images are 512/1024/1536 px
  squares. To rule out a size/format shortcut, everything was also
  center-cropped, resized to 512 px and JPEG-re-encoded identically
  (`scripts/normalize_eval_set.py`) and rescored.
- Scoring: `scripts/compare_detectors.py` (explicit softmax, AI class taken
  from each model's `id2label`) and the production path
  `scripts/evaluate.py`, which agree exactly for the current detector.
- CPU, macOS arm64. Detector latency ~160 ms/image.

## Results (AI = positive class)

| Model | License | ROC-AUC (raw) | ROC-AUC (normalized) | Acc @ 0.65 policy | Real-image recall @ 0.65 |
|---|---|---|---|---|---|
| **Nahrawy/AIorNot** (current) | Apache-2.0 | 0.852 | 0.850 | 0.770 | 0.79 |
| Organika/sdxl-detector | CC-BY-NC-3.0 | 0.884 | 0.879 | 0.775 | 0.69 |
| dima806/ai_vs_real_image_detection | Apache-2.0 | 0.678 | – | 0.600 | 0.31 |
| umm-maybe/AI-image-detector | CC-BY-4.0 | 0.495 (chance) | – | 0.480 | 0.81 |

Current detector under the project's 3-way policy (200 images): 96 "AI
suspected" (78% correct), 68 "Real suspected" (84% correct), 36
"Inconclusive". Bootstrap 95% CI for its AUC: 0.80 to 0.90.

## What this means
- The current detector is **moderately useful, not reliable**: roughly one
  in five real photos is flagged "AI suspected" and one in four AI images
  is missed at the 0.65 policy.
- `sdxl-detector` is nominally higher, but the AUC difference has a 95% CI
  of −0.03 to +0.09 (includes zero) and its license is non-commercial, so
  the detector was **not replaced**.
- Re-picking the threshold on half the data and testing on the other half
  gave no gain over 0.65 (both 0.78), so the thresholds were **not changed**.
- The two other candidates performed poorly or at chance on this data and
  should not be used here.

## Limits
- 200 images from one dataset; the CI above is wide. Different generators,
  JPEG-heavy social-media images, or faces could shift results either way.
- Possible overlap between a model's training data and this dataset is
  unknown for all four models.
- RAG retrieval quality is still **unmeasured** (needs hand-labeled
  `data/eval/rag_eval.jsonl`).

Reproduce:
```bash
python scripts/fetch_eval_data.py date3k2/raw_real_fake_images --split test --per-class 100
python scripts/evaluate.py data/eval_sets/raw_real_fake_images
python scripts/compare_detectors.py data/eval_sets/raw_real_fake_images Nahrawy/AIorNot Organika/sdxl-detector
```


## Research retrieval (added 2026-10-18)
`evaluation/rag_questions.jsonl` holds 56 hand-written questions: 40 answerable (with the expected source
document), 6 on-topic but not covered by the documents, 10 off-topic. `python scripts/export_evaluation.py rag`
runs them through the real hybrid retriever and writes `evaluation/rag_eval.json`.

| Metric | Result |
|---|---|
| Hit@1 / Hit@3 / Hit@5 | 97.5% / 100% / 100% (39/40, 40/40, 40/40) |
| MRR | 0.988 |
| Hit@1, BM25-only / vector-only / hybrid | 0.95 / 0.975 / 0.975 (MRR 0.965 / 0.988 / 0.988): hybrid ties vector-only on this set |
| Off-topic questions refused | 10/10 |
| On-topic but not covered, refused | 3/6 (weakest area) |
| Answers whose every claim is verbatim from its cited passage | 56/56 |

The index used is rebuilt from the built-in documents only (6 documents, 28 passages), so the result does not depend on any local uploads.

Caveats: labels are document-level and were written by someone who had read the documents, so wording overlap
flatters retrieval; 40 questions give wide intervals (see the artifact). The relevance gate was tuned on these
same questions.

## Detector, manual observation
A single image generated with ChatGPT's image model was classified "Real" at 98.5% by the production detector. One
case proves nothing about a rate, but it matches the documented risk that the detector predates modern generators.
