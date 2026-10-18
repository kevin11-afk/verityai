#!/usr/bin/env python3
"""
Research-document ingestion CLI.

Usage:
    python scripts/ingest_documents.py                 # ingest new/changed docs
    python scripts/ingest_documents.py --rebuild        # drop index, re-ingest everything
    python scripts/ingest_documents.py --inspect        # list what's in knowledge_base/
    python scripts/ingest_documents.py --debug "query"  # run a retrieval query and show hits
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.rag.ingestion import load_knowledge_base_directory  # noqa: E402
from src.rag.chunking import chunk_document  # noqa: E402
from src.rag.vector_store import VectorStore  # noqa: E402
from src.rag.retrieval import HybridRetriever  # noqa: E402


def inspect_knowledge_base() -> None:
    docs = load_knowledge_base_directory(config.KNOWLEDGE_BASE_DIR)
    if not docs:
        print(f"No documents found in {config.KNOWLEDGE_BASE_DIR}")
        return
    print(f"Found {len(docs)} document(s):")
    for doc in docs:
        n_pages = len(doc.pages)
        n_chars = len(doc.full_text)
        print(f"  - {doc.source_path.name}: title='{doc.title}', pages={n_pages}, "
              f"chars={n_chars}, source_url={doc.source_url or '(none)'}")


def ingest(rebuild: bool) -> None:
    print(f"Loading documents from {config.KNOWLEDGE_BASE_DIR} ...")
    docs = load_knowledge_base_directory(config.KNOWLEDGE_BASE_DIR)
    if not docs:
        print("No documents found. The app will still run, but the Research Assistant "
              "and evidence citations will have nothing to retrieve. "
              "Add .pdf/.txt/.md files to knowledge_base/ and re-run this script.")

    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  {doc.source_path.name}: {len(chunks)} chunk(s)")

    store = VectorStore()
    t0 = time.perf_counter()
    if rebuild:
        print("Rebuilding index from scratch...")
        store.rebuild(all_chunks)
        n_added = len(all_chunks)
    else:
        n_added = store.add_chunks(all_chunks)
    elapsed = time.perf_counter() - t0

    print(f"Indexed {n_added} new chunk(s) (skipped {len(all_chunks) - n_added} already present) "
          f"in {elapsed:.1f}s.")
    print(f"Vector store now has {store.count()} total chunk(s) at {config.CHROMA_DB_DIR}")


def debug_query(query: str) -> None:
    store = VectorStore()
    if store.count() == 0:
        print("Vector store is empty. Run ingestion first: python scripts/ingest_documents.py")
        return
    retriever = HybridRetriever(store)
    passages = retriever.retrieve(query, top_k=config.RETRIEVAL_TOP_K)
    print(f"Query: {query!r} -> {len(passages)} passage(s)\n")
    for i, p in enumerate(passages, 1):
        print(f"[{i}] {p.title} (p.{p.page_number}) | rrf={p.rrf_score:.4f} "
              f"vec_rank={p.vector_rank} bm25_rank={p.bm25_rank}")
        snippet = p.text.strip().replace("\n", " ")
        print(f"    {snippet[:200]}{'...' if len(snippet) > 200 else ''}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="VerityAI knowledge-base ingestion")
    parser.add_argument("--rebuild", action="store_true", help="Drop and rebuild the whole index")
    parser.add_argument("--inspect", action="store_true", help="List documents found, then exit")
    parser.add_argument("--debug", metavar="QUERY", help="Run a retrieval query and print results")
    args = parser.parse_args()

    if args.inspect:
        inspect_knowledge_base()
        return
    if args.debug:
        debug_query(args.debug)
        return

    ingest(rebuild=args.rebuild)


if __name__ == "__main__":
    main()
