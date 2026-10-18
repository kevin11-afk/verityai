# data/

This directory is for local, untracked data — it is excluded from git via
`.gitignore` (except this file).

- `data/tmp/` — temporary files written during image/video validation and
  video frame extraction. Cleaned up automatically after each analysis;
  safe to delete manually at any time.
- `data/eval/images/real/` and `data/eval/images/ai/` — optional, for
  `scripts/evaluate.py`'s detector evaluation. Populate with a licensed,
  labeled dataset (see README.md "Evaluation instructions" for free
  options) — not provided in this repo.
- `data/eval/rag_eval.jsonl` — optional, for `scripts/evaluate.py`'s RAG
  evaluation. One JSON object per line:
  `{"question": "...", "relevant_chunk_ids": ["..."]}`. Get real chunk IDs
  by running `python scripts/ingest_documents.py --debug "your question"`
  and copying the IDs of passages you've manually judged relevant.
