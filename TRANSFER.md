# VerityAI: handoff and status

Last updated 2026-10-18. This records what was actually run, not what is intended.
Companion docs: `README.md`, `EVALUATION.md`, `PORTFOLIO.md`.

**Bottom line**
- The original Streamlit app is **deployed and working** at https://verityai.streamlit.app (from `main`; verified
  on the live URL: image analysis, report download, research, off-topic refusal).
- A **new product UI (React + TypeScript + FastAPI)** is built and verified locally on branch
  `feature/premium-react-frontend`. It is **not deployed anywhere**, and the free-hosting route for it (a second
  Streamlit Community Cloud app running `product/server.py`) has been tested only locally, including behind a
  proxy that mimics Cloud's `/~/+/` URL prefix. Whether Cloud itself passes the custom routes through is unknown.
- Detector accuracy is moderate and a modern generator was missed in manual testing (see Remaining issues).

## New product UI: what exists
`api/` (FastAPI, typed Pydantic schemas) wraps the existing `src/` code; `web/` is the React app; `product/static`
is its committed build; `evaluation/*.json` are measured artifacts the Evaluation lab reads; `product/server.py`
is the `st.App` entry for Streamlit Cloud; `Dockerfile` and `firebase.json` are alternative hosting configs.

## Verification run (macOS arm64, Python 3.11.15, Node 22)
| Check | Result |
|---|---|
| `pytest tests/` | **80 passed** |
| `python scripts/verify_pipeline.py` (real models) | **26/26** |
| `python scripts/verify_api.py` (real models, every endpoint) | **24/24** |
| `npm run lint` / `typecheck` / `test` / `build` | clean; 8 unit tests |
| `npm audit --omit=dev` | 0 vulnerabilities (7 dev-only advisories remain, all in Tailwind 3's build-time glob chain; the fix is a Tailwind 4 migration, not done) |
| Browser, real API | image + video analysis, report page, Research (answer and refusal), Knowledge (upload, duplicate, search, confirmed removal), Evaluation (slider recomputes), Reports, Methodology, mobile 375 px, drawer, deep-link refresh, server-offline state |
| `product/server.py` under Streamlit 1.66, with and without the `/~/+/` prefix proxy | API, SPA, deep links work |

## Bugs found and fixed while building (all with tests)
1. **Chunker corrupted passages**: its overlap logic duplicated and spliced text (and started chunks mid-word),
   so "verbatim" quotes could be spliced sentences. Replaced with a sentence-aware chunker; chunks are now
   contiguous source text. Persisted indexes carry a chunker version (sidecar file) and rebuild when stale.
2. **Relevance gate refused short keyword questions** ("What is Mean Reciprocal Rank?"). Added a lexical rescue
   (two or more key terms together in one passage); Hit@1 rose from 90% to 97.5% on the labeled set (measured before the later chunker fix; final figures are in EVALUATION.md).
3. **Rank fusion was nondeterministic** (ties broken by Python set order), which changed metrics between runs.
4. Weakly related passages padded result lists; passages supported by neither retriever are now dropped.
5. The API schema silently dropped new diagnostics fields (Pydantic ignores extras); pinned by a test.
6. Extractive answers used headings, fragments and bibliography lines as "claims"; rewritten and tested.
7. An unloadable lazy page chunk blanked the whole app; added an error boundary.
8. Smaller: buttons nested in links, "Ai" capitalisation, Retry shown for non-retryable errors, truncated stats.

## Hosting reality (checked 2026-10-18)
Hugging Face Docker Spaces now require a paid plan to create; Render/Koyeb free tiers (512 MB) cannot hold PyTorch;
Cloud Run needs a billing account. The only ₹0 option with enough RAM that is already proven for this stack is
Streamlit Community Cloud, hence the `st.App` route. Docker and Firebase configs exist but were never executed
(neither Docker nor the Firebase CLI was available here).

## Provenance
Built in a network-less cloud sandbox, exported as a Git bundle, and cloned
to `~/code/verityai`. History: `aa111b2` (scaffold), `25218c2` (first
TRANSFER.md), then the verification and fix commits on branch
`fix/real-env-setup`. The default branch in the bundle was `master`; this
clone renamed it to `main`.

## Earlier verification of the original Streamlit app (before the UI rewrite; superseded by the table above)
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

## Cloud deployment (2026-10-18)
- **Attempt 1 failed:** build log showed `Using Python 3.14.8`. Python had not
  been set to 3.11, and `torch==2.4.1` has no wheel for 3.14 (uv and pip both
  failed). Fix: delete and recreate the app with Advanced settings → Python 3.11.
- **Attempt 2 succeeded** (Python 3.11 selected). I did not see the build log, so
  the torch build (`+cpu`) and final memory use are unconfirmed.
- Verified on the live URL, via the browser pane: home page loads; Research
  Assistant shows 34 chunks (index auto-built on first start) and returns cited
  passages for an on-topic question; the off-topic question gets the
  "insufficient evidence" notice; Image Analysis validated a (canvas-generated)
  JPEG, ran real inference, showed 5 citations and 2 report buttons, and the
  JSON report endpoint returned HTTP 200 with correct content.
- **Not verified on the live URL:** Video Analysis, Evaluation, About, the
  Markdown report, the saved-file step, sustained use or memory limits, and
  sleep/wake behavior after inactivity.
- Note: `curl` to the app URL needs a cookie jar and the health endpoint is not
  informative behind Cloud's proxy; test in a browser instead.

## Remaining issues
1. **The new UI is not deployed.** The `st.App` route on Streamlit Community Cloud is unconfirmed (see README → Hosting). Cloud memory with the full stack (PyTorch + embeddings + FastAPI inside Streamlit) is unmeasured on Linux; locally the old app peaked around 650 MB RSS.
2. **A modern generator fooled the detector** (ChatGPT image scored 98.5% "Real"). The Evaluation lab's numbers come from older generators and do not reflect this. Fixing it properly needs a modern-generator test set and a model with a commercial-friendly licence that does better; none was found or evaluated yet.
3. **Detector accuracy is moderate**: ROC-AUC 0.85 (95% CI 0.80–0.90), ~21% of real photos flagged, ~25% of AI images missed, on 200 images from one dataset. The dataset states no licence (scores only are published).
4. **RAG evaluation is small and self-authored**: 56 questions written by the people who read the documents; document-level labels only; Hit@1 97.5%, Hit@5 100%; 3/6 on-topic-but-uncovered questions are refused. The relevance gate (distance 0.65 plus lexical rescue) was calibrated on this same small set.
5. Answers are extractive (verbatim sentences). They can be on-topic but not the best sentence, and cannot synthesise across sources.
6. History and uploaded documents are lost when a host without persistent storage restarts (disclosed in the UI).
7. No authentication or rate limiting: anyone with the URL can upload files and add or delete knowledge-base documents on a shared instance.
8. Dev-only npm advisories remain (Tailwind 3 build chain, 7 total). `Dockerfile` and `firebase.json` were never run.
9. Not tested: real file-picker uploads (files were injected by script), screen readers, Safari/Firefox, very large files, concurrent users.
10. The Evaluation artifacts were generated against the local data in `data/eval_sets` (git-ignored); regenerating them needs `scripts/fetch_eval_data.py` and `scripts/compare_detectors.py` first.

## Exact next steps
```bash
cd ~/code/verityai && source .venv/bin/activate
pip install -r requirements-api.txt          # adds FastAPI to the existing environment
pytest tests/ -q                             # expect 80 passed
python scripts/verify_api.py                 # real models, expect ALL PASS
uvicorn api.main:app --port 8000             # open http://localhost:8000/product/
cd web && npm install && npm run lint && npm run typecheck && npm test && npm run build
```
Then deploy per README → "Deploy on Streamlit Community Cloud" (main file `product/server.py`, Python 3.11) and
record what the Cloud build log and the live URL actually show before calling it deployed. After that, the highest
value work is a modern-generator evaluation set and detector.
