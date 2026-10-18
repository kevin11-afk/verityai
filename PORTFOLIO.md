# VerityAI — portfolio & handoff notes

Honest status as of this writing. Update the checkboxes as you actually
run and verify each step on your machine — don't flip one to "verified"
without having run it.

## Feature status

| Feature | Code written | Verified running |
|---|---|---|
| Image upload, validation, preview | ✅ | ✅ (run in the live app on macOS, 2026-10-18) |
| Real detector inference (Nahrawy/AIorNot) | ✅ | ✅ (real weights; labels `real`/`ai`; accuracy NOT validated) |
| Decision policy / categories | ✅ | ✅ (logic unit-tested with mocked scores, passed) |
| Downloadable JSON/Markdown report | ✅ | ◑ (buttons render; downloaded file contents not opened) |
| Video frame extraction + per-frame detection | ✅ | ✅ (one generated 4 s MP4; not a real-video test) |
| RAG ingestion (PDF/TXT/MD + chunking + metadata) | ✅ | ✅ (chunking/metadata unit-tested, passed) |
| ChromaDB vector store | ✅ | ✅ (37 chunks, real embeddings) |
| BM25 keyword search | ✅ | ✅ |
| Hybrid retrieval (RRF) | ✅ | ✅ (vector and BM25 ranks both populated; recall not measured) |
| Evidence-grounded explanation templates | ✅ | ✅ (grounding/citation logic unit-tested, passed) |
| 6 seed knowledge-base documents, real citations | ✅ | — (fact-checked against live sources during writing) |
| Streamlit UI — all 6 pages | ✅ | ✅ (all load; Image, Video, Research flows exercised) |
| Tests (`tests/`) | ✅ | ✅ (40 passed, 0 skipped, Python 3.11) |
| Evaluation script | ✅ | ◑ (runs and prints "Not evaluated"; no labeled data) |
| GitHub push | ⬜ | ⬜ |
| Streamlit Community Cloud deployment | ⬜ | ⬜ |

**Why some things say "code written, not verified running":** this
project was built in a cloud sandbox with no network access to PyPI or
Hugging Face Hub (confirmed by diagnostics — DNS resolution failed for
every host except one hardcoded Anthropic API entry). Nothing beyond
`numpy`/`opencv`/`pillow` could be installed there. The plan was to move
to your linked Mac (full internet, matches the real target hardware) to
install dependencies and run everything end-to-end — pick this file back
up once that's done and flip the checkboxes you've actually verified.

**What *was* verified without network**, using a dependency-free manual
test script (`_smoke_test.py`, safe to delete once `pytest` runs for
real): file validation (valid images pass, corrupted/renamed files are
rejected, path traversal is impossible), the detector's decision-policy
thresholds, text chunking and citation-metadata preservation, report
JSON/Markdown generation, and the evidence-grounding logic (no LLM
hallucination path exists — there is no LLM). These are real, run, passing
checks — not claims.

**Update 2026-10-18 (macOS arm64, Python 3.11.15):** the items marked ✅
above were run on a real install. Bugs found and fixed in that pass are
listed in `TRANSFER.md`. **Detector accuracy was first measured on
2026-10-18 on a 200-image sample** (AUC 0.85, ~77% accuracy, ~21% of real
photos flagged as AI): see `EVALUATION.md`. Quote those numbers only with
their caveats; the sample is small and single-dataset.

## Two-minute demo script

1. **(0:00–0:15)** "This is VerityAI — a screening tool for AI-generated
   images and video that backs every result with real research evidence,
   not just a black-box score." Open the Home page.
2. **(0:15–0:45)** Go to Image Analysis. Upload a clearly AI-generated
   image (e.g. something you know came from Midjourney/SD). Point out:
   the real model score, the fixed decision policy text, and the cited
   research passage underneath (not generated text — a retrieved quote
   with a source).
3. **(0:45–1:05)** Upload a real photo. Point out the "Real suspected" or
   "Inconclusive" result and that the app explicitly does not claim this
   proves authenticity.
4. **(1:05–1:30)** Switch to Research Assistant. Ask: "Why do detectors
   struggle with new generators?" Show the cited passage and source link.
5. **(1:30–1:50)** Open the downloaded Markdown report from step 2/3 —
   show that every field from the spec (model name, scores, policy,
   citations, limitations, timestamp) is present.
6. **(1:50–2:00)** "Everything here is free and open-source — no paid
   APIs anywhere in the pipeline." Close on the About page's architecture
   diagram.

## Technical interview explanation

*"Walk me through how it works."*

> VerityAI is two independent systems wired together, deliberately kept
> separate. First, a pretrained Swin-Tiny image classifier
> (`Nahrawy/AIorNot`, Apache-2.0) produces the actual AI/Real prediction —
> that's the only thing in the whole app that decides 'is this AI or
> not.' Second, a hybrid RAG pipeline — BM25 keyword search plus a
> ChromaDB vector index over Sentence-Transformer embeddings, fused with
> Reciprocal Rank Fusion — retrieves relevant passages from a small
> hand-curated research knowledge base. A deterministic template then
> combines the two: it states the detector's own output as a structural
> fact, and separately quotes-and-cites whatever research passages were
> retrieved. There's no LLM anywhere, which means there's no hallucination
> risk in the explanation step — every claim either comes from the
> model's own scores (computed in code) or is a direct quote with a
> citation. The interesting engineering constraint was the hosting
> budget: Streamlit Community Cloud's free tier gives about 1GB RAM, so
> the model choice was driven as much by parameter count and license as
> by accuracy — see `knowledge_base/04_detector_model_cards.md` for the
> comparison I did against three other candidates.

*"What would you do differently with more time?"*

> Convert the detector to ONNX Runtime to cut the PyTorch import/RAM
> overhead, which is the actual bottleneck on the free hosting tier, not
> the model's own size. I'd also want a second, independently-trained
> detector to cross-check against — a single classifier's errors are
> correlated with its own training data's blind spots, which is exactly
> the generalization problem documented in the knowledge base.

## What's NOT done / known gaps

- Not yet pushed to GitHub or deployed (see README "Deployment").
- Full `pytest` run with real dependencies pending (sandbox network
  constraint — see status table above).
- `scripts/evaluate.py` has not been run against real labeled data; no
  accuracy/retrieval numbers are claimed anywhere in this repo pending
  that (the Evaluation Dashboard will correctly show "Not evaluated"
  until you supply `data/eval/`).

## Portfolio summary

> VerityAI is a multimodal AI-generated media screening application that
> combines pretrained computer vision models with hybrid Retrieval-
> Augmented Generation. It produces image predictions, experimental video
> frame analysis, evidence-grounded explanations, and research citations
> while explicitly communicating model uncertainty.

(Quoted as specified — use once the checkboxes above are actually
checked, not before.)
