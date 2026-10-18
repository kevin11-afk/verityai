# Evaluation methodology used by VerityAI's scripts/evaluate.py

## Detector metrics

- **Precision** = TP / (TP + FP): of everything flagged "AI," what
  fraction actually was AI. Low precision means many real images get
  wrongly flagged.
- **Recall** = TP / (TP + FN): of everything that actually was AI, what
  fraction got flagged. Low recall means AI images slip through
  undetected.
- **F1** = harmonic mean of precision and recall; a single number that
  penalizes models that only do well on one of the two.
- **Confusion matrix**: the raw [[TN, FP], [FN, TP]] counts behind the
  above — always worth checking directly, since precision/recall/F1 can
  look reasonable even when one class is tiny.
- **ROC-AUC**: how well the model's raw score ranks AI images above real
  images across *all* possible thresholds, not just the one VerityAI
  happens to use. Useful for judging the underlying model independent of
  the specific decision policy in src/config.py.
- **Inference latency**: wall-clock time per image; relevant for whether
  the model fits a hosting platform's request-timeout limits.

## Why test data must be separate from threshold-tuning data

VerityAI's decision thresholds (`AI_SUSPECTED_THRESHOLD`,
`REAL_SUSPECTED_THRESHOLD` in `src/config.py`) are fixed, documented
defaults — they were **not** tuned against any evaluation set bundled
with this project. If thresholds were tuned on the same data used to
report accuracy, the reported metrics would be optimistic and would not
predict performance on new, unseen images. Anyone adjusting these
thresholds for their own evaluation data should hold out a separate test
set that was not used for tuning, and report metrics only on that held-out
set.

## RAG metrics

- **Recall@K**: for a question with a known relevant passage, whether
  that passage appears anywhere in the top-K retrieved results. Measures
  whether the retriever found the right needle at all, regardless of its
  exact rank.
- **Mean Reciprocal Rank (MRR)**: the average of 1/rank of the first
  relevant passage across all test questions. Rewards finding the right
  passage near the top, not just somewhere in the top-K.
- **Citation coverage**: the fraction of questions for which the pipeline
  returned at least one passage to cite at all (as opposed to correctly
  returning the "insufficient evidence" notice, or — the failure case —
  silently returning nothing).
- **Groundedness (basic check)**: whether every claim in a generated
  explanation can be traced back to a specific retrieved passage. Because
  VerityAI's generation step (`src/rag/generation.py`) only ever quotes
  and cites retrieved text rather than paraphrasing freely with an LLM,
  groundedness here is close to automatic by construction — the main
  failure mode to check for is a citation attached to the wrong passage,
  not an invented claim.
- **Retrieval failure analysis**: manually inspecting the cases where
  Recall@K missed, to see whether the failure was a vocabulary mismatch
  (BM25 would have needed a keyword overlap that wasn't there — hybrid
  retrieval is specifically meant to reduce this failure mode) or a
  genuine knowledge-base gap (the answer just isn't in any document yet).

"Not evaluated" is printed by `scripts/evaluate.py` wherever the relevant
labeled data isn't present, rather than a fabricated number — see
README "Evaluation instructions" for how to supply real evaluation data.
