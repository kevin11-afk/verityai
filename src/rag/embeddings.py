"""
Sentence-embedding wrapper around Sentence Transformers.

Lazily imported and cached per-process, same rationale as the detector:
keep startup fast when embeddings aren't needed yet, and never hold two
copies of a ~90MB model in memory.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np

from src import config


class EmbeddingUnavailableError(RuntimeError):
    """Raised when sentence-transformers / the embedding model can't be loaded."""


@lru_cache(maxsize=1)
def _load_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingUnavailableError(
            "sentence-transformers is not installed. Run: pip install -r requirements.txt"
        ) from exc
    try:
        return SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    except Exception as exc:  # noqa: BLE001
        raise EmbeddingUnavailableError(
            f"Could not load embedding model '{config.EMBEDDING_MODEL_NAME}': {exc}"
        ) from exc


def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = _load_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])[0]
