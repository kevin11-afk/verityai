"""
ChromaDB-backed persistent vector store for knowledge-base chunks.

A local, on-disk ChromaDB instance — no database server, no network
dependency once the embedding model is cached, and no paid vector DB.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from src import config
from src.rag.chunking import CHUNKER_VERSION, Chunk
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

    # The chunker version lives in a sidecar file: ChromaDB overwrites collection metadata when
    # an existing collection is reopened, so metadata cannot be trusted to record how it was built.
    def _version_file(self) -> Path:
        return Path(self._persist_directory) / f"chunker_version_{self._collection_name}.txt"

    def _write_version(self) -> None:
        self._version_file().write_text(CHUNKER_VERSION, encoding="utf-8")

    def stale(self) -> bool:
        """True when a non-empty index was built by an older chunker and must be rebuilt."""
        if self.count() == 0:
            return False
        try:
            return self._version_file().read_text(encoding="utf-8").strip() != CHUNKER_VERSION
        except OSError:
            return True  # no record: built before versioning existed

    def rebuild(self, chunks: List[Chunk]) -> None:
        """Drop and recreate the collection, then index `chunks` from scratch.
        Used by scripts/ingest_documents.py --rebuild."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name, metadata={"hnsw:space": "cosine"}
        )
        self.add_chunks(chunks)
        self._write_version()

    def add_chunks(self, chunks: List[Chunk]) -> int:
        """Add chunks, skipping any chunk_id already present (duplicate-safe
        re-ingestion of the same document)."""
        if not chunks:
            return 0
        was_empty = self.count() == 0

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
        if was_empty:
            self._write_version()
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

    def list_sources(self) -> dict:
        """{source_file: {"title", "chunks", "source_url"}} for every indexed document."""
        data = self.get_all_documents_with_metadata()
        out: dict = {}
        for meta in data.get("metadatas", []):
            entry = out.setdefault(
                meta["source_file"],
                {"title": meta.get("title", ""), "chunks": 0, "source_url": meta.get("source_url") or None},
            )
            entry["chunks"] += 1
        return out

    def delete_source(self, source_file: str) -> int:
        """Remove every chunk of one document. Returns how many were removed."""
        existing = self._collection.get(where={"source_file": source_file})
        ids = existing.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)
