# Knowledge base

These six Markdown files are **original summaries written for this
project**, not reproductions of any paper's text. Every factual claim
cites a real, publicly available paper, dataset, or specification by
title, authors, venue, year, and URL — nothing here is an invented
citation.

They were written (and fact-checked against each source) specifically so
VerityAI ships with *something* real to retrieve and cite out of the box,
without redistributing copyrighted PDFs of the underlying papers.

## Adding more documents

Drop any `.pdf`, `.txt`, or `.md` file into this directory and run:

```bash
python scripts/ingest_documents.py
```

- PDFs are parsed with PyMuPDF; page numbers are preserved automatically.
- To attach a canonical source URL to a document for citation purposes,
  add a sidecar file named `<same-filename>.url` containing just the URL,
  e.g. `corvi2023.pdf.url` next to `corvi2023.pdf`.
- Before adding a real paper PDF, check that its license/repository terms
  allow you to store and redistribute a copy (many publishers do not).
  arXiv preprints are generally safe to download for personal/educational
  use but check the specific paper's license before bundling a copy into
  a public GitHub repo — when in doubt, add a `.url` sidecar pointing to
  the official page and summarize the content yourself instead of storing
  the PDF, exactly as this starter knowledge base does.

## Rebuilding the index from scratch

```bash
python scripts/ingest_documents.py --rebuild
```

## Inspecting what's indexed

```bash
python scripts/ingest_documents.py --inspect
python scripts/ingest_documents.py --debug "does detector accuracy generalize to new generators?"
```
