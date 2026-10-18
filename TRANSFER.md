# VerityAI — transfer notes

Written 2026-10-18 while packaging the project for continued development
with Claude Code on macOS. This file records what is actually known to
work, not what is intended to work. See `PORTFOLIO.md` for the feature
checklist and `README.md` for full setup instructions.

## Where this package came from

- The project was built in a cloud sandbox at `/home/claude/verityai`
  that had no access to PyPI or Hugging Face Hub, so the heavy
  dependencies were never installed there.
- The sandbox's export (`verityai_project.tar.gz` plus
  `verityai_git_history.bundle`) was then copied to
  `~/Documents/` on the Mac and extracted to `~/Documents/verityai`.
- This package is built from the **Git bundle**, which has the full
  history (one commit, `aa111b2`, branch `master`) and the same files as
  the tarball.
- The extracted copy at `~/Documents/verityai` is **missing
  `tests/test_validation.py`** and has **no `.git` directory of its own**.
  It sits inside a Git repository rooted at the home directory, which
  belongs to an unrelated project. Use this package rather than that copy.

## Project status (honest summary)

| Area | Status |
|---|---|
| Source code (app, 5 pages, `src/`, scripts) | Written. All files compile under Python 3.12. |
| Unit tests | **30 passed, 1 skipped** on the Mac (Python 3.12.4, pytest 9.1.1) |
| `_smoke_test.py` | **12/12 passed** on the Mac |
| Retrieval tests (`tests/test_retrieval.py`) | **Skipped, not run**: `chromadb` and `rank_bm25` aren't installed |
| Real model inference | **Not run anywhere yet** |
| `streamlit run app.py` end-to-end | **Not run yet** |
| Knowledge-base ingestion (`scripts/ingest_documents.py`) | **Not run yet** |
| Evaluation (`scripts/evaluate.py`) | **Not run**: no labeled data exists |
| GitHub push | Not done |
| Streamlit Community Cloud deployment | **Not deployed** |

The application is **not fully tested** and **not deployed**.

## Completed features (code written)

- Image upload with validation (extension, magic bytes, size, dimensions)
  and path-traversal-safe temp files. Unit-tested.
- Detector wrapper with a fixed decision policy (`P ≥ 0.65` → suspected,
  otherwise Inconclusive). The policy logic is unit-tested against
  mocked model outputs.
- Experimental video analysis: OpenCV frame sampling (≤8 frames, ≤30 s)
  and mean aggregation of per-frame image scores.
- RAG pipeline: PDF/TXT/MD ingestion, chunking with citation metadata,
  ChromaDB vector store, BM25, Reciprocal Rank Fusion, and deterministic
  template explanations. There is no LLM anywhere.
- JSON/Markdown report generation. Unit-tested.
- Streamlit UI: Home, Image Analysis, Video Analysis, Research
  Assistant, Evaluation, and About.
- 6 knowledge-base documents in `knowledge_base/`.

## Untested features

- Real detector inference: the `transformers` pipeline loading
  `Nahrawy/AIorNot`.
- Embeddings (`sentence-transformers`), the ChromaDB store, BM25, and
  hybrid retrieval against a real index.
- PDF ingestion (`pymupdf`).
- Every Streamlit page in a running app, including the report download
  buttons.
- Video analysis on a real `.mp4` or `.mov` file.
- The evaluation script and dashboard.
- Behaviour within Streamlit Cloud's ~1 GB RAM limit.

## Known errors and risks

No failing tests are currently known. These are open risks, not
confirmed bugs:

1. **Detector label names are unverified.** `src/detector/image_detector.py`
   lowercases the pipeline labels and looks for `ai` and `real`. If the
   `Nahrawy/AIorNot` model config uses other label names (for example
   `LABEL_0`/`LABEL_1`), the scores fall back to 0 and every result
   becomes "Inconclusive" without any error. Check
   `model.config.id2label` on the first real run.
2. **Python version mismatch.** The README targets Python 3.10–3.11. The
   Mac has 3.12.4, and the pinned `torch==2.4.1`, `chromadb==0.5.5`, and
   `numpy==1.26.4` may not all have 3.12 wheels or work together on it.
   Using a 3.11 venv is the safer choice.
3. **Mixed system packages.** The Mac's system Python already has `torch`
   and `streamlit` at unknown versions that may not match the pins.
   Always work inside a fresh venv.
4. **Deployment RAM.** PyTorch, Transformers, Sentence-Transformers, and
   ChromaDB together may exceed Streamlit Cloud's free-tier RAM limit.
5. **No pre-built index.** `chroma_db/` is gitignored, so a deployment
   must either build the index at startup or commit one (see README
   "Deployment", step 7).

## Selected detector

**`Nahrawy/AIorNot`**: a Swin-Tiny fine-tune (~28M params, Apache-2.0)
trained on the "aiornot" dataset. It was chosen over
`umm-maybe/AI-image-detector`, `Organika/sdxl-detector`, and
`dima806/ai_vs_real_image_detection` for its license and small size.
See `knowledge_base/04_detector_model_cards.md` and `src/config.py`. You
can override it with the `VERITYAI_DETECTOR_MODEL` environment variable.
Known limitation: its training data predates 2024.

Embeddings model: `sentence-transformers/all-MiniLM-L6-v2`.

## What was excluded from this package

Virtual environments, `__pycache__`, `.pytest_cache`, Hugging Face model
caches and weights, `chroma_db/`, `data/tmp/`, `data/uploads/`, and
`.streamlit/secrets.toml`. None of these existed in the repo, and no
secrets were found in the source.

## Exact next steps (on the Mac)

```bash
# 0. Unpack (or clone from the bundle) into a clean folder
mkdir -p ~/code && cd ~/code
unzip ~/Documents/verityai_transfer/verityai_transfer.zip   # creates ./verityai with .git
#   or: git clone ~/Documents/verityai_transfer/verityai.bundle verityai
cd verityai
git log --oneline            # expect the scaffold commit + the transfer commit

# 1. Python 3.11 venv (install with: brew install python@3.11)
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.4.1      # on macOS the default wheel is CPU/MPS; the README's --index-url .../cpu line is for Linux
pip install -r requirements.txt

# 2. Full test suite, which should now include test_retrieval.py
pytest tests/ -v

# 3. Check the detector's label names (risk #1)
python -c "from transformers import AutoConfig; print(AutoConfig.from_pretrained('Nahrawy/AIorNot').id2label)"

# 4. Build the knowledge-base index
python scripts/ingest_documents.py
python scripts/ingest_documents.py --debug "why do detectors fail on new generators"

# 5. Run the app and test each page by hand
streamlit run app.py
```

Then:

6. Update the "Verified running" column in `PORTFOLIO.md` for each item
   you actually verified.
7. Optionally delete `_smoke_test.py` once `pytest` covers the same checks.
8. Create a GitHub repo and push. The default branch is `master`; rename
   it to `main` with `git branch -m main` if you follow the README's
   deployment steps.
9. Deploy to Streamlit Community Cloud using README "Deployment", then
   test the public URL before calling it deployed.
10. Optionally, add labeled data to `data/eval/` and run
    `python scripts/evaluate.py`.

---

## Update 2026-10-18: first real-environment verification

Environment: macOS arm64, Python 3.11.15 (via `uv venv --python 3.11`),
a fresh venv installed from `requirements.txt`, all free/local, ₹0.

**Verified by running it**
- Clean-room install from `requirements.txt` alone, then `pytest`: **40 passed, 0 skipped** (retrieval tests now run).
- `Nahrawy/AIorNot` downloaded and loaded; `id2label = {0: 'real', 1: 'ai'}`, so risk #1 above is cleared.
- Real inference on three sample photos, a noise image, and an 8-frame video sample.
- `ingest_documents.py`: 37 chunks indexed with real embeddings; ChromaDB + BM25 + RRF return results with both ranks populated.
- Streamlit app: Home, About, Evaluation, Video, Image, and Research Assistant pages loaded without exceptions. Image Analysis, Video Analysis, and Research Assistant were exercised through the browser. Files were injected into the uploader by script rather than with a real file picker.
- `scripts/evaluate.py` runs and correctly prints "Not evaluated".

**Bugs found and fixed (branch `fix/real-env-setup`)**
1. The pipeline applied an independent **sigmoid**, not softmax, so scores didn't sum to 1 and contradicted the documented policy. Fixed with `function_to_apply="softmax"` in `src/detector/image_detector.py`. Decision-policy outcomes change as a result.
2. `st.image(use_container_width=...)` isn't supported by pinned Streamlit 1.39, which crashed Image Analysis on every upload. Now `use_column_width`.
3. BM25 filter `score > 0` dropped real keyword matches on small corpora, because rank_bm25's IDF is ≤ 0 for terms in about half the documents. Now filters on token overlap.
4. A retrieval test was non-deterministic: it used Python's per-process `hash()` for fake embeddings. Now uses `zlib.crc32`.
5. chromadb 0.5.5 telemetry broke with posthog ≥ 6 and logged errors on every call. Pinned `posthog==5.4.0` and disabled telemetry.
6. A fresh deploy had no index (`chroma_db/` is gitignored). Added `src/rag/bootstrap.py`, which builds it on first start. It has a unit test.
7. Added `--extra-index-url` for CPU torch to `requirements.txt` for Linux hosts.

**Still incomplete or unverified**
- **Detector accuracy is unmeasured and looks poor.** Two real sample photos were labeled "AI-generated suspected" (0.70, 0.97) and a noise image 0.65. Treat the detector as unvalidated. Run `scripts/evaluate.py` on a labeled set (e.g. CIFAKE, with its caveats) before making any claim, and consider whether the 0.65 thresholds or the model need to change.
- Retrieval quality is unmeasured (no `rag_eval.jsonl`).
- Downloaded JSON/Markdown report files were not opened (the buttons render).
- Video was tested only on a synthetic MP4 built from still photos.
- Streamlit Community Cloud: **not deployed, and the deployment config is untested there.** Unverified: the CPU torch wheel being selected, the RAM limit (peak ~540 MB locally for models, excluding Streamlit), and cold-start time.
- Cosmetic: retrieved quotes show raw Markdown `#` headings; video explanation text says "this image".
- A secondary `pytest` DeprecationWarning from chromadb/pydantic (harmless).
