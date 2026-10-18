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
    vector_distance: Optional[float] = None  # cosine distance; None if not in the vector top-k


@dataclass
class RetrievalDiagnostics:
    """What the retriever did for one query: used by the API to explain results
    and abstentions (nothing here is invented; every field is measured)."""
    query: str
    corpus_chunks: int
    best_vector_distance: Optional[float]
    distance_threshold: float
    abstained: bool
    abstain_reason: Optional[str]
    gate_passed_by: Optional[str]  # "vector" | "lexical" | None when abstained
    lexical_coverage: Optional[float]
    vector_candidates: int
    bm25_candidates: int
    top_k: int
    weak_filtered: int = 0  # fused candidates dropped because neither retriever supported them


_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    "a an the of to in on for and or is are was were be been being it its this that these those "
    "with as by at from how what why when which who whom do does did can could should would will "
    "i you we they he she them their there than then so if not no yes about into over under more "
    "most such very also".split()
)
# A short query can sit far from long passages in embedding space even when its
# key terms occur verbatim in a passage. If at least MIN_LEXICAL_TERMS content
# terms of the query appear together in one chunk, and they cover at least
# LEXICAL_COVERAGE of the query's content terms, the relevance gate is passed.
MIN_LEXICAL_TERMS = 2
LEXICAL_COVERAGE = 0.75


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

    def _lexical_coverage(self, query: str) -> Optional[float]:
        """Best fraction of the query's content terms found together in a single chunk,
        or None when the query has fewer than MIN_LEXICAL_TERMS content terms."""
        terms = {t for t in _tokenize(query) if t not in STOPWORDS and len(t) > 2}
        if len(terms) < MIN_LEXICAL_TERMS:
            return None
        best = max((len(terms & tokens) for tokens in self._bm25_token_sets), default=0)
        return best / len(terms)

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

    def retrieve(self, query: str, top_k: int = config.RETRIEVAL_TOP_K) -> List[EvidencePassage]:
        return self.retrieve_with_diagnostics(query, top_k)[0]

    def retrieve_with_diagnostics(
        self, query: str, top_k: int = config.RETRIEVAL_TOP_K
    ) -> "tuple[List[EvidencePassage], RetrievalDiagnostics]":
        fetch_k = max(top_k * 3, 10)  # overfetch each retriever before fusing
        n_chunks = self._store.count()
        if n_chunks == 0:
            return [], RetrievalDiagnostics(
                query, 0, None, config.MAX_BEST_VECTOR_DISTANCE, True,
                "The knowledge base is empty.", None, None, 0, 0, top_k,
            )
        vec_results = self._store.query(query, top_k=fetch_k)
        vec_ids_list = vec_results.get("ids", [[]])[0]
        distances = vec_results.get("distances", [[]])[0]
        best = min(distances) if distances else None
        coverage = self._lexical_coverage(query)
        gate = None
        if vec_ids_list and best is not None and best <= config.MAX_BEST_VECTOR_DISTANCE:
            gate = "vector"
        elif coverage is not None and coverage >= LEXICAL_COVERAGE:
            gate = "lexical"
        if gate is None:
            # Outside the knowledge base: let callers report insufficient evidence.
            dist_txt = f"{best:.3f}" if best is not None else "n/a"
            return [], RetrievalDiagnostics(
                query, n_chunks, best, config.MAX_BEST_VECTOR_DISTANCE, True,
                f"Best vector match (distance {dist_txt}) is farther than the relevance "
                f"threshold ({config.MAX_BEST_VECTOR_DISTANCE:.2f}), and no passage contains the "
                "query's key terms together.",
                None, coverage, len(vec_ids_list), 0, top_k,
            )
        vector_ids = vec_ids_list
        bm25_ids = self._bm25_ranked_ids(query, fetch_k)

        vector_rank = {cid: i + 1 for i, cid in enumerate(vector_ids)}
        bm25_rank = {cid: i + 1 for i, cid in enumerate(bm25_ids)}

        # Deterministic order (vector hits first, then BM25-only hits): ties in the fused score
        # must not depend on Python's per-process set ordering.
        all_ids = list(dict.fromkeys([*vector_ids, *bm25_ids]))
        k = config.RRF_K_CONSTANT

        fused = []
        for cid in all_ids:
            score = 0.0
            if cid in vector_rank:
                score += 1.0 / (k + vector_rank[cid])
            if cid in bm25_rank:
                score += 1.0 / (k + bm25_rank[cid])
            fused.append((cid, score))

        best_rank = {cid: min(vector_rank.get(cid, 10**6), bm25_rank.get(cid, 10**6)) for cid in all_ids}
        fused.sort(key=lambda x: (-x[1], best_rank[x[0]]))
        # Drop passages that neither retriever supports well: far in vector space AND no keyword
        # match. The best match already passed the gate; this stops weak neighbours padding the list.
        distance_of = dict(zip(vec_ids_list, distances))
        def supported(cid: str) -> bool:
            d = distance_of.get(cid)
            return cid in bm25_rank or (d is not None and d <= config.MAX_BEST_VECTOR_DISTANCE)
        kept = [item for item in fused if supported(item[0])]
        weak_filtered = min(len(fused), top_k) - min(len(kept), top_k)
        top_ids = kept[:top_k] if kept else fused[:1]

        # Need text/metadata for ids that only came from the vector retriever's
        # fetch, which already has it; fall back to the store for anything else.
        passages = []
        vec_docs = vec_results.get("documents", [[]])[0]
        vec_metas = vec_results.get("metadatas", [[]])[0]
        vector_result_cache = {
            cid: (doc, meta) for cid, doc, meta in zip(vec_ids_list, vec_docs, vec_metas)
        }
        distance_by_id = dict(zip(vec_ids_list, distances))

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
                    vector_distance=distance_by_id.get(cid),
                )
            )
        diagnostics = RetrievalDiagnostics(
            query, n_chunks, best, config.MAX_BEST_VECTOR_DISTANCE, False, None, gate, coverage,
            len(vector_ids), len(bm25_ids), top_k, weak_filtered,
        )
        return passages, diagnostics
