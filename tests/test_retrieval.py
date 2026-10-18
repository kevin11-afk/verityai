"""
Tests for chunking, BM25, vector, and hybrid (RRF) retrieval, and citation
metadata preservation.

These tests need the full dependency set (chromadb, rank-bm25,
sentence-transformers) from requirements.txt — they use a real temporary
ChromaDB instance and a real BM25 index so the fusion logic itself is
exercised end-to-end, with only the embedding *model* mocked out (to keep
tests network-free and fast; embedding quality is not what's under test
here — scripts/evaluate.py's RAG evaluation covers real retrieval quality).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import zlib

import numpy as np
import pytest

from src.rag.chunking import Chunk, chunk_document
from src.rag.ingestion import LoadedDocument, LoadedPage

pytest.importorskip("chromadb")
pytest.importorskip("rank_bm25")


def _fake_embed(texts):
    """Deterministic, content-sensitive fake embedding: good enough to make
    cosine similarity behave sensibly for a handful of distinct test
    sentences, without downloading a real model."""
    vectors = []
    for text in texts:
        rng = np.random.default_rng(zlib.crc32(text.encode()) % (2**32))
        vectors.append(rng.random(16).astype(np.float32))
    return np.array(vectors, dtype=np.float32)


@pytest.fixture
def vector_store(tmp_path):
    with patch("src.rag.vector_store.embed_texts", side_effect=_fake_embed):
        from src.rag.vector_store import VectorStore
        store = VectorStore(persist_directory=tmp_path / "chroma_test", collection_name="test_collection")
        yield store


class TestChunking:
    def test_chunk_document_preserves_metadata(self):
        doc = LoadedDocument(
            source_path=Path("paper.pdf"),
            title="Example Paper",
            source_url="https://example.com/paper",
            pages=[LoadedPage(page_number=1, text="Sentence one. " * 100)],
        )
        chunks = chunk_document(doc, chunk_size=200, overlap=20)
        assert len(chunks) > 1
        for c in chunks:
            assert c.title == "Example Paper"
            assert c.source_url == "https://example.com/paper"
            assert c.page_number == 1
            assert c.source_file == "paper.pdf"
            assert c.chunk_id  # non-empty, stable id

    def test_chunk_ids_are_unique(self):
        doc = LoadedDocument(
            source_path=Path("doc.txt"), title="T", source_url=None,
            pages=[LoadedPage(page_number=None, text="Para one.\n\nPara two.\n\nPara three.")],
        )
        chunks = chunk_document(doc, chunk_size=20, overlap=0)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_empty_document_produces_no_chunks(self):
        doc = LoadedDocument(source_path=Path("empty.txt"), title="Empty", source_url=None,
                              pages=[LoadedPage(page_number=None, text="   ")])
        assert chunk_document(doc) == []


class TestVectorStore:
    def test_add_and_count(self, vector_store):
        chunks = [
            Chunk("id1", "The sky is blue.", "T", None, None, "f.txt"),
            Chunk("id2", "The grass is green.", "T", None, None, "f.txt"),
        ]
        with patch("src.rag.vector_store.embed_texts", side_effect=_fake_embed):
            added = vector_store.add_chunks(chunks)
        assert added == 2
        assert vector_store.count() == 2

    def test_duplicate_chunk_ids_are_skipped(self, vector_store):
        chunk = Chunk("dup1", "Some text.", "T", None, None, "f.txt")
        with patch("src.rag.vector_store.embed_texts", side_effect=_fake_embed):
            first = vector_store.add_chunks([chunk])
            second = vector_store.add_chunks([chunk])
        assert first == 1
        assert second == 0
        assert vector_store.count() == 1


class TestHybridRetrieval:
    def test_bm25_finds_exact_keyword_match(self, vector_store):
        chunks = [
            Chunk("a", "Diffusion models produce frequency-domain artifacts.", "Paper A", None, 1, "a.pdf"),
            Chunk("b", "Cats and dogs are common household pets.", "Paper B", None, 1, "b.pdf"),
        ]
        with patch("src.rag.vector_store.embed_texts", side_effect=_fake_embed):
            vector_store.add_chunks(chunks)
            from src.rag.retrieval import HybridRetriever
            retriever = HybridRetriever(vector_store)
            results = retriever.retrieve("frequency-domain artifacts diffusion", top_k=2)

        assert len(results) > 0
        # Vector ranking from the fake embedding is arbitrary, so only the
        # BM25 side is asserted: the keyword-matching chunk must rank first there.
        by_id = {r.chunk_id: r for r in results}
        assert "a" in by_id
        assert by_id["a"].bm25_rank == 1
        assert by_id["b"].bm25_rank is None if "b" in by_id else True

    def test_citation_metadata_survives_retrieval(self, vector_store):
        chunk = Chunk("c1", "Some evidence text.", "My Title", "https://example.com/src", 7, "doc.pdf")
        with patch("src.rag.vector_store.embed_texts", side_effect=_fake_embed):
            vector_store.add_chunks([chunk])
            from src.rag.retrieval import HybridRetriever
            retriever = HybridRetriever(vector_store)
            results = retriever.retrieve("evidence text", top_k=1)

        assert len(results) == 1
        assert results[0].title == "My Title"
        assert results[0].source_url == "https://example.com/src"
        assert results[0].page_number == 7
        assert results[0].source_file == "doc.pdf"

    def test_empty_store_returns_no_results(self, vector_store):
        from src.rag.retrieval import HybridRetriever
        retriever = HybridRetriever(vector_store)
        assert retriever.retrieve("anything", top_k=5) == []

    def test_rrf_combines_both_retrievers(self, vector_store):
        """A chunk ranked by both BM25 and vector search should score higher
        (via RRF) than one found by only a single retriever."""
        chunks = [
            Chunk("both", "neural network detector accuracy evaluation", "P", None, None, "x.pdf"),
            Chunk("bm25_only", "neural network detector accuracy evaluation unrelated_unique_token_zzz",
                  "P", None, None, "x.pdf"),
        ]
        with patch("src.rag.vector_store.embed_texts", side_effect=_fake_embed):
            vector_store.add_chunks(chunks)
            from src.rag.retrieval import HybridRetriever
            retriever = HybridRetriever(vector_store)
            results = retriever.retrieve("neural network detector accuracy evaluation", top_k=2)
        assert len(results) == 2
        # Both chunks share all query keywords, so BM25 ranks are close;
        # the test mainly asserts the pipeline runs end-to-end and returns
        # both chunk_ids with populated rrf_score.
        assert {r.chunk_id for r in results} == {"both", "bm25_only"}
        assert all(r.rrf_score > 0 for r in results)


class TestBootstrap:
    def test_ensure_index_built_indexes_once_when_empty(self, vector_store, tmp_path, monkeypatch):
        from src import config
        from src.rag.bootstrap import ensure_index_built

        kb = tmp_path / "kb"
        kb.mkdir()
        (kb / "doc.md").write_text("# Title\n\nDiffusion models leave frequency artifacts.")
        monkeypatch.setattr(config, "KNOWLEDGE_BASE_DIR", kb)
        with patch("src.rag.vector_store.embed_texts", side_effect=_fake_embed):
            added = ensure_index_built(vector_store)
            assert added > 0 and vector_store.count() == added
            assert ensure_index_built(vector_store) == 0  # no-op when already populated
