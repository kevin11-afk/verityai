"""
Hybrid retrieval: BM25 keyword search + ChromaDB vector search, combined
with Reciprocal Rank Fusion (RRF).

RRF is used (rather than, say, a learned re-ranker) because it needs no
training data of its own and is a standard, well-understood way to combine
rankings from two retrievers with different score scales.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from src import config
from src.rag.vector_store import VectorStore


@dataclass
class EvidencePassage:
    chunk_id: str
    text: str
    title: str
    source_file: str
    page_number: Optional[int]
    source_url: Optional[str]
    vector_rank: Optional[int]
    bm25_rank: Optional[int]
    rrf_score: float


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class HybridRetriever:
    def __init__(self, store: VectorStore):
        self._store = store
        self._bm25 = None
        self._bm25_chunk_ids: List[str] = []
        self._bm25_metadata = {}
        self._bm25_texts = {}
        self._bm25_token_sets: List[set] = []
        self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        from rank_bm25 import BM25Okapi

        data = self._store.get_all_documents_with_metadata()
        ids = data.get("ids", [])
        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])

        self._bm25_chunk_ids = ids
        self._bm25_metadata = {cid: meta for cid, meta in zip(ids, metadatas)}
        self._bm25_texts = {cid: doc for cid, doc in zip(ids, documents)}

        if not documents:
            self._bm25 = None
            return

        tokenized_corpus = [_tokenize(doc) for doc in documents]
        self._bm25_token_sets = [set(t) for t in tokenized_corpus]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def refresh(self) -> None:
        """Rebuild the in-memory BM25 index after the vector store changes."""
        self._build_bm25_index()

    def _bm25_ranked_ids(self, query: str, top_k: int) -> List[str]:
        if self._bm25 is None:
            return []
        query_tokens = set(_tokenize(query))
        scores = self._bm25.get_scores(list(query_tokens))
        # Require real token overlap rather than score > 0: BM25Okapi's IDF is
        # 0 (or negative) for terms present in about half the corpus or more,
        # which would wrongly discard genuine keyword matches in small corpora.
        candidates = [
            (cid, score)
            for cid, score, tokens in zip(self._bm25_chunk_ids, scores, self._bm25_token_sets)
            if query_tokens & tokens
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in candidates[:top_k]]

    def _vector_ranked_ids(self, query: str, top_k: int) -> List[str]:
        if self._store.count() == 0:
            return []
        results = self._store.query(query, top_k=top_k)
        ids = results.get("ids", [[]])
        return ids[0] if ids else []

    def retrieve(self, query: str, top_k: int = config.RETRIEVAL_TOP_K) -> List[EvidencePassage]:
        fetch_k = max(top_k * 3, 10)  # overfetch each retriever before fusing
        vector_ids = self._vector_ranked_ids(query, fetch_k)
        bm25_ids = self._bm25_ranked_ids(query, fetch_k)

        vector_rank = {cid: i + 1 for i, cid in enumerate(vector_ids)}
        bm25_rank = {cid: i + 1 for i, cid in enumerate(bm25_ids)}

        all_ids = set(vector_ids) | set(bm25_ids)
        k = config.RRF_K_CONSTANT

        fused = []
        for cid in all_ids:
            score = 0.0
            if cid in vector_rank:
                score += 1.0 / (k + vector_rank[cid])
            if cid in bm25_rank:
                score += 1.0 / (k + bm25_rank[cid])
            fused.append((cid, score))

        fused.sort(key=lambda x: x[1], reverse=True)
        top_ids = fused[:top_k]

        # Need text/metadata for ids that only came from the vector retriever's
        # fetch, which already has it; fall back to the store for anything else.
        passages = []
        vector_result_cache = {}
        if vector_ids:
            vec_results = self._store.query(query, top_k=fetch_k)
            vec_docs = vec_results.get("documents", [[]])[0]
            vec_metas = vec_results.get("metadatas", [[]])[0]
            vec_ids_list = vec_results.get("ids", [[]])[0]
            vector_result_cache = {
                cid: (doc, meta) for cid, doc, meta in zip(vec_ids_list, vec_docs, vec_metas)
            }

        for cid, rrf_score in top_ids:
            if cid in vector_result_cache:
                text, meta = vector_result_cache[cid]
            else:
                text = self._bm25_texts.get(cid, "")
                meta = self._bm25_metadata.get(cid, {})

            page_number = meta.get("page_number", -1)
            passages.append(
                EvidencePassage(
                    chunk_id=cid,
                    text=text,
                    title=meta.get("title", "Unknown source"),
                    source_file=meta.get("source_file", ""),
                    page_number=None if page_number == -1 else page_number,
                    source_url=meta.get("source_url") or None,
                    vector_rank=vector_rank.get(cid),
                    bm25_rank=bm25_rank.get(cid),
                    rrf_score=rrf_score,
                )
            )
        return passages
