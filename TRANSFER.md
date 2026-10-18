# VerityAI: handoff and status

Last updated 2026-10-18. This records what was actually run, not what is
intended. Companion docs: `README.md` (setup, deployment), `EVALUATION.md`
(detector accuracy), `PORTFOLIO.md` (feature checklist).

**Bottom line:** the app works end to end locally on macOS. It is **not
deployed**, the Streamlit Cloud steps have never been run, and detector
accuracy is only moderate. It is ready to attempt a free deployment, with the
caveats in "Remaining issues".

## Provenance
Built in a network-less cloud sandbox, exported as a Git bundle, and cloned
to `~/code/verityai`. History: `aa111b2` (scaffold), `25218c2` (first
TRANSFER.md), then the verification and fix commits on branch
`fix/real-env-setup`. The default branch in the bundle was `master`; this
clone renamed it to `main`.

## Test results (macOS arm64, Python 3.11.15, fresh `uv` venv)
| Check | Result |
|---|---|
| `pytest tests/` | **42 passed, 0 skipped** (includes real ChromaDB and BM25 retrieval) |
| `python _smoke_test.py` | 12/12 passed |
| Clean-room install from `requirements.txt` alone, then `pytest` | 40/40 at that time (before 2 later tests were added) |
| `python scripts/verify_pipeline.py` (real models and real index) | **26/26 checks passed** |
| Linux dependency resolution (`uv pip compile`, manylinux, py3.11) | Resolves; `torch==2.4.1+cpu`, no CUDA packages |

`verify_pipeline.py` checks:
- The detector's real labels are `{0: 'real', 1: 'ai'}` and its scores sum to 1.
- All 6 knowledge-base files are indexed (34 chunks).
- 5 topical queries find the right source file in the top 5, with both BM25
  and vector ranks populated.
- Every retrieved passage's text really occurs in its claimed source file.
- Citations equal the retrieved passages, in order.
- Quoted findings come from the retrieved text.
- The detector summary reports exactly the model's scores and category.
- Empty evidence yields an explicit notice, with no citations.
- An off-topic query returns no passages.

## App workflows exercised in the browser pane
Run against `streamlit run app.py`. Files were injected into the uploader by
script, not through a native file picker.
- **Image Analysis:** upload, validation, real inference, scores, cited
  explanation, and both report downloads. The JSON and Markdown report
  endpoints returned HTTP 200 with correct content and filenames. The
  browser pane doesn't save files to disk, so the saved-file step is untested.
- **Invalid upload** (a text file named `.jpg`): rejected with a clear
  message and no crash. This was a bug that is now fixed.
- **Research Assistant:** an on-topic question returned cited passages from
  the right file. An off-topic question returned the explicit
  "insufficient evidence" message.
- **Video Analysis (experimental):** one generated 4 s MP4 produced 8 sampled
  frames, an aggregate result, citations, and report buttons. Not tested on a
  real-world video.
- Home, About and Evaluation pages load without exceptions.

## Bugs found and fixed on `fix/real-env-setup`
1. The pipeline applied sigmoid instead of the documented softmax, so scores didn't sum to 1. Now explicit softmax.
2. `st.image(use_container_width=...)` isn't supported by pinned Streamlit 1.39 and crashed every upload.
3. The image preview was drawn before validation, so an invalid file crashed the page.
4. BM25 `score > 0` dropped real matches on small corpora. Now filters on token overlap.
5. The retriever queried the vector store twice. Now once.
6. Off-topic questions still returned five cited passages. Added a relevance gate (`MAX_BEST_VECTOR_DISTANCE`, default 0.65, env `VERITYAI_MAX_DISTANCE`).
7. `knowledge_base/README.md` was indexed and could be cited as a source. Now excluded.
8. A flaky retrieval test used Python's per-process `hash()`. Now `zlib.crc32`.
9. chromadb telemetry broke with posthog ≥ 6 and logged errors. Pinned `posthog==5.4.0` and disabled telemetry.
10. A fresh deploy had no index. `src/rag/bootstrap.py` builds it on first start.
11. Embeddings auto-selected Apple MPS (about 1 GB of extra GPU memory locally). Now forced to CPU, matching the hosting tier.
12. Added the CPU torch `--extra-index-url` to `requirements.txt`.
13. The README claimed an environment variable to disable video that doesn't exist. Removed the claim.

## Git hygiene (verified)
Across all history, no images, videos, model weights, databases, caches,
virtualenvs, `data/tmp` or eval files, `secrets.toml` or key-like strings
were ever committed. 50 tracked files, the largest 16 KB. `.venv/`,
`chroma_db/`, `data/tmp/` and `data/eval*/` are git-ignored and exist only
locally.

## Detector
`Nahrawy/AIorNot` (Swin-Tiny, Apache-2.0), kept after comparing three
alternatives (`EVALUATION.md`). On 200 images from one dataset: ROC-AUC 0.85,
~77% accuracy, ~21% of real photos flagged as AI, ~25% of AI images missed at
the 0.65 policy. Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (CPU).

## Memory and requirements vs. the free tier
- Streamlit docs (checked 2026-10-18): 690 MB to 2.7 GB per app, subject to change.
- Measured locally after image, research and video in one process: ~650 MB RSS,
  ~390 MB peak footprint. **Linux was not measured.** At the 690 MB floor the
  headroom is thin.
- Cloud installs `requirements.txt` with uv, which resolved the CPU torch
  wheel in my Linux dry run. The actual Cloud build has not been run.
- Budget: nothing here uses a paid API, paid hosting, or any billing account.

## Remaining issues
1. **Not deployed.** The GitHub push and the Streamlit Cloud build and run are untested.
2. **Linux and Cloud memory unmeasured.**
3. **Detector accuracy is moderate.** The evaluation sample is small (200 images) and covers one dataset.
4. **Retrieval quality unmeasured.** The relevance gate was calibrated on only 18 queries against a 6-document knowledge base.
5. Saved report files weren't opened from disk; only the served content was checked.
6. Video was tested on one synthetic clip only.
7. Cosmetic: retrieved quotes can show raw Markdown `#` headings; the video explanation text says "this image"; pytest shows chromadb/pydantic deprecation warnings.
8. The dataset used for evaluation states no license. It is kept local and git-ignored.

## Exact next steps
```bash
cd ~/code/verityai
source .venv/bin/activate            # or: uv venv --python 3.11 .venv && uv pip install -r requirements.txt
pytest tests/ -v                      # expect 42 passed
python scripts/verify_pipeline.py     # expect ALL PASS (needs network on first run)
streamlit run app.py
```
Then follow **README → Deployment** step by step. After the first Cloud
deploy, record the build log's torch version, the app's memory behavior, and
the results of the step 5 checks here before claiming it is deployed.
