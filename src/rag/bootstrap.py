"""Build the knowledge-base index on first start when it is empty.

chroma_db/ is git-ignored, so a fresh deployment (e.g. Streamlit Community
Cloud) starts with no index. Pages call this from their cached retriever
loaders so the index is built once per process; it is a no-op when the index
already exists.
"""
from __future__ import annotations

from src import config
from src.rag.chunking import chunk_document
from src.rag.ingestion import load_knowledge_base_directory
from src.rag.vector_store import VectorStore


def ensure_index_built(store: VectorStore) -> int:
    """Index knowledge_base/ if the store is empty. Returns chunks added."""
    if store.count() > 0:
        return 0
    chunks = []
    for doc in load_knowledge_base_directory(config.KNOWLEDGE_BASE_DIR):
        chunks.extend(chunk_document(doc))
    return store.add_chunks(chunks) if chunks else 0
