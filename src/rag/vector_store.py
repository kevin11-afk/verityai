"""
ChromaDB-backed persistent vector store for knowledge-base chunks.

A local, on-disk ChromaDB instance — no database server, no network
dependency once the embedding model is cached, and no paid vector DB.
"""
from __future__ import annotations

from typing import List, Optional

from src import config
from src.rag.chunking import Chunk
from src.rag.embeddings import embed_texts


class VectorStore:
    def __init__(self, persist_directory=None, collection_name: Optional[str] = None):
        import os

        # chromadb 0.5.x's telemetry client is incompatible with current posthog
        # and logs errors on every call; telemetry is off for privacy anyway.
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
        import chromadb

        self._persist_directory = str(persist_directory or config.CHROMA_DB_DIR)
        self._client = chromadb.PersistentClient(
            path=self._persist_directory,
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self._collection_name = collection_name or config.CHROMA_COLLECTION_NAME
        # cosine distance matches normalized sentence-embedding similarity
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name, metadata={"hnsw:space": "cosine"}
        )

    def rebuild(self, chunks: List[Chunk]) -> None:
        """Drop and recreate the collection, then index `chunks` from scratch.
        Used by scripts/ingest_documents.py --rebuild."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name, metadata={"hnsw:space": "cosine"}
        )
        self.add_chunks(chunks)

    def add_chunks(self, chunks: List[Chunk]) -> int:
        """Add chunks, skipping any chunk_id already present (duplicate-safe
        re-ingestion of the same document)."""
        if not chunks:
            return 0

        existing_ids = set()
        try:
            existing = self._collection.get(ids=[c.chunk_id for c in chunks])
            existing_ids = set(existing.get("ids", []))
        except Exception:  # noqa: BLE001 - treat lookup failure as "nothing existing"
            pass

        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
        if not new_chunks:
            return 0

        embeddings = embed_texts([c.text for c in new_chunks])
        self._collection.add(
            ids=[c.chunk_id for c in new_chunks],
            embeddings=embeddings.tolist(),
            documents=[c.text for c in new_chunks],
            metadatas=[
                {
                    "title": c.title,
                    "source_file": c.source_file,
                    "page_number": c.page_number if c.page_number is not None else -1,
                    "source_url": c.source_url or "",
                }
                for c in new_chunks
            ],
        )
        return len(new_chunks)

    def count(self) -> int:
        return self._collection.count()

    def query(self, query_text: str, top_k: int):
        query_embedding = embed_texts([query_text])[0]
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, max(self.count(), 1)),
        )
        return results

    def get_all_documents_with_metadata(self):
        """Used by BM25 index building — returns every chunk's text + metadata."""
        if self.count() == 0:
            return {"ids": [], "documents": [], "metadatas": []}
        return self._collection.get(include=["documents", "metadatas"])
