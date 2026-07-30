"""
Hybrid retriever combining dense (semantic) and sparse (keyword) search.

Supports:
  - BM25 (sparse) retrieval for keyword matching
  - Vector (dense) retrieval for semantic matching
  - Hybrid fusion with reciprocal rank fusion (RRF)
  - Weighted score combination
  - Metadata filtering
  - Batch retrieval
"""

from __future__ import annotations

import asyncio
import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np


class RetrievalMode(str, Enum):
    """Available retrieval modes."""

    DENSE = "dense"           # Vector similarity only
    SPARSE = "sparse"         # BM25 keyword only
    HYBRID = "hybrid"         # Combined (RRF or weighted)
    HYBRID_RRF = "hybrid_rrf" # Reciprocal rank fusion
    HYBRID_WEIGHTED = "hybrid_weighted"  # Weighted score combination


@dataclass
class RetrievalResult:
    """A single retrieval result."""

    chunk_id: str
    text: str
    score: float
    rank: int = 0
    dense_score: float = 0.0
    sparse_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_document: Optional[str] = None


@dataclass
class RetrievalResponse:
    """Complete retrieval response."""

    query: str
    results: List[RetrievalResult]
    total_candidates: int = 0
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────
# BM25 (Sparse) Retriever
# ──────────────────────────────────────────────────────────────────


class BM25Retriever:
    """BM25 sparse retrieval for keyword-based search.

    Implements Okapi BM25 scoring for document retrieval.
    Supports incremental indexing and batch querying.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        """Initialize the BM25 retriever.

        Args:
            k1: Term frequency saturation parameter.
            b: Length normalization parameter.
            epsilon: Smoothing parameter for IDF.
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        # Indexed data
        self._documents: List[str] = []
        self._doc_ids: List[str] = []
        self._doc_lengths: List[int] = []
        self._avg_doc_length: float = 0.0

        # Term statistics
        self._term_to_doc_freq: Dict[str, int] = defaultdict(int)
        self._doc_term_freqs: List[Dict[str, int]] = []
        self._total_docs = 0

        # Precomputed IDF
        self._idf: Dict[str, float] = {}

    def index(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        reset: bool = True,
    ) -> None:
        """Index a list of documents.

        Args:
            documents: List of document texts.
            doc_ids: Optional list of document IDs.
            metadata: Optional metadata for each document.
            reset: Whether to reset existing index.
        """
        if reset:
            self.reset()

        if doc_ids is None:
            doc_ids = [str(i) for i in range(len(documents))]

        for i, (doc, doc_id) in enumerate(zip(documents, doc_ids)):
            terms = self._tokenize(doc)
            term_freqs: Dict[str, int] = defaultdict(int)
            for term in terms:
                term_freqs[term] += 1

            self._documents.append(doc)
            self._doc_ids.append(doc_id)
            self._doc_lengths.append(len(terms))
            self._doc_term_freqs.append(dict(term_freqs))
            self._total_docs += 1

            # Update document frequencies
            for term in set(terms):
                self._term_to_doc_freq[term] += 1

        # Recompute average length and IDF
        self._avg_doc_length = (
            sum(self._doc_lengths) / len(self._doc_lengths)
            if self._doc_lengths
            else 0.0
        )
        self._compute_idf()

    def add_document(
        self,
        text: str,
        doc_id: Optional[str] = None,
    ) -> str:
        """Add a single document to the index.

        Args:
            text: Document text.
            doc_id: Document ID (generated if None).

        Returns:
            Document ID.
        """
        if doc_id is None:
            doc_id = str(len(self._documents))

        terms = self._tokenize(text)
        term_freqs: Dict[str, int] = defaultdict(int)
        for term in terms:
            term_freqs[term] += 1

        self._documents.append(text)
        self._doc_ids.append(doc_id)
        self._doc_lengths.append(len(terms))
        self._doc_term_freqs.append(dict(term_freqs))
        self._total_docs += 1

        for term in set(terms):
            self._term_to_doc_freq[term] += 1

        self._avg_doc_length = sum(self._doc_lengths) / len(self._doc_lengths)
        self._compute_idf()

        return doc_id

    def search(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.0,
    ) -> List[Tuple[int, float]]:
        """Search for documents matching the query.

        Args:
            query: Search query.
            top_k: Number of results to return.
            score_threshold: Minimum score threshold.

        Returns:
            List of (doc_index, score) tuples, sorted by score descending.
        """
        if not self._documents:
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        scores = np.zeros(self._total_docs)

        for term in query_terms:
            idf = self._idf.get(term, 0.0)
            if idf == 0.0:
                continue

            for doc_idx in range(self._total_docs):
                tf = self._doc_term_freqs[doc_idx].get(term, 0)
                if tf == 0:
                    continue

                doc_len = self._doc_lengths[doc_idx]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_len / max(self._avg_doc_length, 1)
                )
                scores[doc_idx] += idf * numerator / denominator

        # Get top-k results
        if score_threshold > 0:
            indices = np.where(scores >= score_threshold)[0]
            top_indices = indices[np.argsort(scores[indices])[::-1][:top_k]]
        else:
            top_indices = np.argsort(scores)[::-1][:top_k]

        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    def reset(self) -> None:
        """Clear the index."""
        self._documents.clear()
        self._doc_ids.clear()
        self._doc_lengths.clear()
        self._doc_term_freqs.clear()
        self._term_to_doc_freq.clear()
        self._idf.clear()
        self._total_docs = 0
        self._avg_doc_length = 0.0

    def _compute_idf(self) -> None:
        """Compute inverse document frequency for all terms."""
        self._idf.clear()
        for term, doc_freq in self._term_to_doc_freq.items():
            self._idf[term] = np.log(
                1 + (self._total_docs - doc_freq + 0.5) / (doc_freq + 0.5)
            )

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple tokenization by lowercase + split.

        Args:
            text: Input text.

        Returns:
            List of tokens.
        """
        import re
        return re.findall(r'\w+', text.lower())


# ──────────────────────────────────────────────────────────────────
# Hybrid Retriever
# ──────────────────────────────────────────────────────────────────


class HybridRetriever:
    """Combined dense + sparse retriever with fusion strategies.

    Orchestrates both BM25 and vector search, combining results
    using reciprocal rank fusion or weighted score combination.
    """

    def __init__(
        self,
        embedder: Any = None,  # Embedder instance
        bm25: Optional[BM25Retriever] = None,
        mode: RetrievalMode = RetrievalMode.HYBRID_RRF,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rrf_k: int = 60,
        top_k: int = 10,
    ):
        """Initialize the hybrid retriever.

        Args:
            embedder: Embedder instance for dense retrieval.
            bm25: BM25 retriever for sparse retrieval.
            mode: Retrieval mode.
            dense_weight: Weight for dense scores (weighted mode).
            sparse_weight: Weight for sparse scores (weighted mode).
            rrf_k: RRF constant.
            top_k: Default number of results.
        """
        self.embedder = embedder
        self.bm25 = bm25 or BM25Retriever()
        self.mode = mode
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k
        self.top_k = top_k

        # Vector storage
        self._documents: List[str] = []
        self._doc_ids: List[str] = []
        self._embeddings: Optional[np.ndarray] = None
        self._metadata: List[Dict[str, Any]] = []

    # ── Indexing ──────────────────────────────────────────────────

    async def index_documents(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Index documents for both dense and sparse retrieval.

        Args:
            documents: List of document texts.
            doc_ids: Optional document IDs.
            metadata: Optional metadata list.
        """
        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(documents))]

        # Store documents
        self._documents = list(documents)
        self._doc_ids = list(doc_ids)
        self._metadata = metadata or [{} for _ in documents]

        # Index in BM25
        self.bm25.index(documents, doc_ids)

        # Generate embeddings
        if self.embedder is not None:
            result = await self.embedder.embed_documents(documents)
            self._embeddings = np.array(result, dtype=np.float32)
        else:
            # Fallback: use zero embeddings (dense search disabled)
            self._embeddings = None

    # ── Retrieval ─────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        mode: Optional[RetrievalMode] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResponse:
        """Retrieve relevant documents for a query.

        Args:
            query: Search query.
            top_k: Number of results.
            mode: Override retrieval mode.
            filters: Optional metadata filters.

        Returns:
            RetrievalResponse with ranked results.
        """
        import time
        start = time.monotonic()

        top_k = top_k or self.top_k
        mode = mode or self.mode

        if mode == RetrievalMode.DENSE:
            results = await self._dense_retrieve(query, top_k, filters)
        elif mode == RetrievalMode.SPARSE:
            results = self._sparse_retrieve(query, top_k, filters)
        elif mode == RetrievalMode.HYBRID_RRF:
            results = await self._hybrid_rrf_retrieve(query, top_k, filters)
        elif mode == RetrievalMode.HYBRID_WEIGHTED:
            results = await self._hybrid_weighted_retrieve(query, top_k, filters)
        else:  # HYBRID default
            results = await self._hybrid_rrf_retrieve(query, top_k, filters)

        latency = (time.monotonic() - start) * 1000

        return RetrievalResponse(
            query=query,
            results=results,
            total_candidates=len(self._documents),
            retrieval_mode=mode,
            latency_ms=latency,
        )

    async def _dense_retrieve(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Dense (vector) retrieval.

        Args:
            query: Search query.
            top_k: Number of results.
            filters: Metadata filters.

        Returns:
            List of RetrievalResult objects.
        """
        if self._embeddings is None or self.embedder is None:
            return []

        query_embedding = np.array(await self.embedder.embed_query(query))

        # Cosine similarity
        similarities = np.dot(self._embeddings, query_embedding) / (
            np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
        )

        return self._collect_results(
            similarities,
            top_k,
            filters,
            dense=True,
        )

    def _sparse_retrieve(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Sparse (BM25) retrieval.

        Args:
            query: Search query.
            top_k: Number of results.
            filters: Metadata filters.

        Returns:
            List of RetrievalResult objects.
        """
        raw_results = self.bm25.search(query, top_k=top_k * 2)

        results = []
        rank = 0
        for doc_idx, score in raw_results:
            if self._apply_filters(doc_idx, filters):
                results.append(RetrievalResult(
                    chunk_id=self._doc_ids[doc_idx],
                    text=self._documents[doc_idx][:500],
                    score=score,
                    rank=rank,
                    sparse_score=score,
                    dense_score=0.0,
                    metadata=self._metadata[doc_idx] if doc_idx < len(self._metadata) else {},
                ))
                rank += 1
                if len(results) >= top_k:
                    break
        return results

    async def _hybrid_rrf_retrieve(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Hybrid retrieval with Reciprocal Rank Fusion.

        Args:
            query: Search query.
            top_k: Number of results.
            filters: Metadata filters.

        Returns:
            List of RetrievalResult objects.
        """
        # Get dense results (more candidates for fusion)
        dense_results = await self._dense_retrieve(query, top_k * 2, filters)
        sparse_results = self._sparse_retrieve(query, top_k * 2, filters)

        # Build rank maps
        dense_rank: Dict[str, int] = {}
        for r in dense_results:
            dense_rank[r.chunk_id] = r.rank + 1  # 1-indexed for RRF

        sparse_rank: Dict[str, int] = {}
        for r in sparse_results:
            sparse_rank[r.chunk_id] = r.rank + 1

        # Fuse scores
        all_ids = set(dense_rank.keys()) | set(sparse_rank.keys())
        fused: List[Tuple[float, str]] = []
        dense_scores: Dict[str, float] = {}
        sparse_scores: Dict[str, float] = {}

        for cid in all_ids:
            dr = dense_rank.get(cid, len(all_ids) + 1)
            sr = sparse_rank.get(cid, len(all_ids) + 1)

            rrf_score = 1.0 / (self.rrf_k + dr) + 1.0 / (self.rrf_k + sr)

            # Find original scores
            for r in dense_results:
                if r.chunk_id == cid:
                    dense_scores[cid] = r.dense_score
                    break
            for r in sparse_results:
                if r.chunk_id == cid:
                    sparse_scores[cid] = r.sparse_score
                    break

            fused.append((rrf_score, cid))

        fused.sort(key=lambda x: x[0], reverse=True)
        fused = fused[:top_k]

        # Build result objects
        results = []
        for rank, (score, cid) in enumerate(fused):
            doc_idx = self._doc_ids.index(cid) if cid in self._doc_ids else -1
            results.append(RetrievalResult(
                chunk_id=cid,
                text=self._documents[doc_idx][:500] if doc_idx >= 0 else "",
                score=score,
                rank=rank,
                dense_score=dense_scores.get(cid, 0.0),
                sparse_score=sparse_scores.get(cid, 0.0),
                metadata=self._metadata[doc_idx] if 0 <= doc_idx < len(self._metadata) else {},
            ))

        return results

    async def _hybrid_weighted_retrieve(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Hybrid retrieval with weighted score combination.

        Args:
            query: Search query.
            top_k: Number of results.
            filters: Metadata filters.

        Returns:
            List of RetrievalResult objects.
        """
        dense_results = await self._dense_retrieve(query, top_k * 2, filters)
        sparse_results = self._sparse_retrieve(query, top_k * 2, filters)

        # Normalize scores
        dense_scores: Dict[str, float] = {}
        sparse_scores: Dict[str, float] = {}

        max_dense = max((r.dense_score for r in dense_results), default=1.0)
        max_sparse = max((r.sparse_score for r in sparse_results), default=1.0)

        for r in dense_results:
            dense_scores[r.chunk_id] = r.dense_score / max(max_dense, 1e-8)
        for r in sparse_results:
            sparse_scores[r.chunk_id] = r.sparse_score / max(max_sparse, 1e-8)

        # Combine weighted scores
        all_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
        combined = []

        for cid in all_ids:
            ds = dense_scores.get(cid, 0.0)
            ss = sparse_scores.get(cid, 0.0)
            weighted = self.dense_weight * ds + self.sparse_weight * ss
            combined.append((weighted, cid, ds, ss))

        combined.sort(key=lambda x: x[0], reverse=True)
        combined = combined[:top_k]

        results = []
        for rank, (score, cid, ds, ss) in enumerate(combined):
            doc_idx = self._doc_ids.index(cid) if cid in self._doc_ids else -1
            results.append(RetrievalResult(
                chunk_id=cid,
                text=self._documents[doc_idx][:500] if doc_idx >= 0 else "",
                score=score,
                rank=rank,
                dense_score=ds,
                sparse_score=ss,
                metadata=self._metadata[doc_idx] if 0 <= doc_idx < len(self._metadata) else {},
            ))

        return results

    def _collect_results(
        self,
        scores: np.ndarray,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        dense: bool = True,
        sparse_scores: Optional[np.ndarray] = None,
    ) -> List[RetrievalResult]:
        """Collect and rank results from score arrays.

        Args:
            scores: Score array.
            top_k: Number of results.
            filters: Metadata filters.
            dense: Whether these are dense scores.
            sparse_scores: Optional sparse scores.

        Returns:
            List of RetrievalResult objects.
        """
        indices = np.argsort(scores)[::-1]
        results = []
        rank = 0

        for idx in indices:
            if scores[idx] <= 0:
                continue
            if not self._apply_filters(idx, filters):
                continue

            results.append(RetrievalResult(
                chunk_id=self._doc_ids[idx],
                text=self._documents[idx][:500] if idx < len(self._documents) else "",
                score=scores[idx],
                rank=rank,
                dense_score=scores[idx] if dense else 0.0,
                sparse_score=0.0 if dense else scores[idx],
                metadata=self._metadata[idx] if idx < len(self._metadata) else {},
            ))
            rank += 1
            if len(results) >= top_k:
                break

        return results

    def _apply_filters(
        self,
        doc_idx: int,
        filters: Optional[Dict[str, Any]],
    ) -> bool:
        """Check if document passes metadata filters.

        Args:
            doc_idx: Document index.
            filters: Metadata filter dict.

        Returns:
            True if document passes all filters.
        """
        if filters is None:
            return True
        if doc_idx >= len(self._metadata):
            return True

        meta = self._metadata[doc_idx]
        for key, value in filters.items():
            if key not in meta:
                return False
            if isinstance(value, list):
                if meta[key] not in value:
                    return False
            elif meta[key] != value:
                return False
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics.

        Returns:
            Dict with index stats.
        """
        return {
            "indexed_documents": len(self._documents),
            "has_embeddings": self._embeddings is not None,
            "embedding_dim": (
                self._embeddings.shape[1]
                if self._embeddings is not None
                else 0
            ),
            "bm25_docs": self.bm25._total_docs,
            "mode": self.mode.value,
        }
