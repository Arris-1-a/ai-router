"""
Re-ranking module for improving retrieval quality.

Supports multiple re-ranking strategies:
  - Cross-encoder re-ranking (most accurate)
  - LLM-based re-ranking (GPT judge)
  - Diversity re-ranking (MMR)
  - Score-based re-ranking
  - Multi-stage pipeline
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class RerankStrategy(str, Enum):
    """Available re-ranking strategies."""

    CROSS_ENCODER = "cross_encoder"
    LLM_JUDGE = "llm_judge"
    MMR = "mmr"                     # Maximal Marginal Relevance
    SCORE_COMBINATION = "score_combination"
    DIVERSITY = "diversity"
    RECENCY = "recency"


@dataclass
class RerankResult:
    """A single re-ranked result."""

    chunk_id: str
    text: str
    score: float
    original_rank: int
    new_rank: int
    original_score: float = 0.0
    rerank_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankResponse:
    """Complete re-ranking response."""

    query: str
    results: List[RerankResult]
    strategy: RerankStrategy
    latency_ms: float = 0.0
    input_count: int = 0
    output_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────
# Base Reranker
# ──────────────────────────────────────────────────────────────────


class Reranker(ABC):
    """Abstract base class for re-rankers."""

    def __init__(
        self,
        top_k: int = 10,
        min_score: float = 0.0,
        strategy: RerankStrategy = RerankStrategy.SCORE_COMBINATION,
    ):
        """Initialize the reranker.

        Args:
            top_k: Number of results to return after re-ranking.
            min_score: Minimum score threshold.
            strategy: Re-ranking strategy.
        """
        self.top_k = top_k
        self.min_score = min_score
        self.strategy = strategy

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> RerankResponse:
        """Re-rank documents based on relevance to query.

        Args:
            query: Search query.
            documents: List of document dicts with 'chunk_id', 'text', 'score'.
            top_k: Override number of top results.

        Returns:
            RerankResponse with re-ranked results.
        """
        ...


# ──────────────────────────────────────────────────────────────────
# Score-Based Reranker
# ──────────────────────────────────────────────────────────────────


class ScoreReranker(Reranker):
    """Simple score-based reranker — sorts by existing scores.

    Useful as a baseline and for score normalization.
    """

    def __init__(
        self,
        top_k: int = 10,
        min_score: float = 0.0,
        score_normalization: str = "minmax",  # "minmax", "softmax", "none"
        **kwargs: Any,
    ):
        """Initialize the score reranker.

        Args:
            top_k: Number of results to return.
            min_score: Minimum score.
            score_normalization: Normalization method.
            **kwargs: Additional arguments.
        """
        super().__init__(
            top_k=top_k,
            min_score=min_score,
            strategy=RerankStrategy.SCORE_COMBINATION,
        )
        self.score_normalization = score_normalization

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> RerankResponse:
        """Re-rank by scores.

        Args:
            query: Search query (unused in score-based).
            documents: Document list.
            top_k: Number of top results.

        Returns:
            RerankResponse.
        """
        start = time.monotonic()
        top_k = top_k or self.top_k

        if not documents:
            return RerankResponse(
                query=query,
                results=[],
                strategy=self.strategy,
                input_count=0,
                output_count=0,
            )

        # Extract scores
        scores = np.array([d.get("score", 0.0) for d in documents])

        # Normalize
        normalized = self._normalize_scores(scores)

        # Sort
        indexed = list(enumerate(normalized))
        indexed.sort(key=lambda x: x[1], reverse=True)

        results = []
        for new_rank, (orig_idx, score) in enumerate(indexed[:top_k]):
            doc = documents[orig_idx]
            if score < self.min_score:
                continue
            results.append(RerankResult(
                chunk_id=doc.get("chunk_id", f"doc_{orig_idx}"),
                text=doc.get("text", "")[:500],
                score=score,
                original_rank=orig_idx,
                new_rank=new_rank,
                original_score=doc.get("score", 0.0),
                rerank_score=score,
                metadata=doc.get("metadata", {}),
            ))

        latency = (time.monotonic() - start) * 1000

        return RerankResponse(
            query=query,
            results=results,
            strategy=self.strategy,
            latency_ms=latency,
            input_count=len(documents),
            output_count=len(results),
        )

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores.

        Args:
            scores: Raw score array.

        Returns:
            Normalized scores.
        """
        if self.score_normalization == "minmax":
            min_val = scores.min()
            max_val = scores.max()
            if max_val - min_val > 0:
                return (scores - min_val) / (max_val - min_val)
            return np.ones_like(scores)
        elif self.score_normalization == "softmax":
            exp_scores = np.exp(scores - scores.max())
            return exp_scores / exp_scores.sum()
        else:  # none
            return scores.copy()


# ──────────────────────────────────────────────────────────────────
# MMR (Maximal Marginal Relevance) Reranker
# ──────────────────────────────────────────────────────────────────


class MMRReranker(Reranker):
    """MMR-based reranker for result diversity.

    Balances relevance to the query with diversity among results,
    reducing redundancy in the top-k results.
    """

    def __init__(
        self,
        embedder: Any = None,
        top_k: int = 10,
        lambda_param: float = 0.7,  # Relevance vs diversity trade-off
        **kwargs: Any,
    ):
        """Initialize the MMR reranker.

        Args:
            embedder: Embedder for computing document similarity.
            top_k: Number of results.
            lambda_param: Balance between relevance (1) and diversity (0).
            **kwargs: Additional arguments.
        """
        super().__init__(top_k=top_k, strategy=RerankStrategy.MMR)
        self.embedder = embedder
        self.lambda_param = lambda_param

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> RerankResponse:
        """Re-rank using MMR for diversity.

        Args:
            query: Search query.
            documents: Document list.
            top_k: Number of results.

        Returns:
            RerankResponse with diverse results.
        """
        start = time.monotonic()
        top_k = top_k or self.top_k

        if not documents or top_k <= 0:
            return RerankResponse(
                query=query,
                results=[],
                strategy=self.strategy,
                input_count=len(documents),
                output_count=0,
            )

        # Get embeddings for all documents
        doc_texts = [d.get("text", "") for d in documents]
        doc_embeddings = await self._get_embeddings(doc_texts)

        scores = np.array([d.get("score", 0.0) for d in documents])

        # Normalize scores
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            scores = np.ones_like(scores)

        # MMR greedy selection
        selected_indices = []
        remaining = list(range(len(documents)))

        for _ in range(min(top_k, len(documents))):
            if not remaining:
                break

            mmr_scores = []
            for idx in remaining:
                relevance = scores[idx]

                # Diversity penalty: max similarity to already selected
                diversity = 0.0
                if selected_indices and doc_embeddings is not None:
                    sims = []
                    for sel_idx in selected_indices:
                        sim = self._cosine_similarity(
                            doc_embeddings[idx],
                            doc_embeddings[sel_idx],
                        )
                        sims.append(sim)
                    diversity = max(sims) if sims else 0.0

                mmr = self.lambda_param * relevance - (1 - self.lambda_param) * diversity
                mmr_scores.append((mmr, idx))

            mmr_scores.sort(key=lambda x: x[0], reverse=True)
            best_idx = mmr_scores[0][1]
            selected_indices.append(best_idx)
            remaining.remove(best_idx)

        # Build results
        results = []
        for new_rank, orig_idx in enumerate(selected_indices):
            doc = documents[orig_idx]
            results.append(RerankResult(
                chunk_id=doc.get("chunk_id", f"doc_{orig_idx}"),
                text=doc.get("text", "")[:500],
                score=float(scores[orig_idx]),
                original_rank=orig_idx,
                new_rank=new_rank,
                original_score=doc.get("score", 0.0),
                rerank_score=float(scores[orig_idx]),
                metadata=doc.get("metadata", {}),
            ))

        latency = (time.monotonic() - start) * 1000

        return RerankResponse(
            query=query,
            results=results,
            strategy=self.strategy,
            latency_ms=latency,
            input_count=len(documents),
            output_count=len(results),
        )

    async def _get_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """Get embeddings for texts.

        Args:
            texts: Text list.

        Returns:
            Embedding array or None.
        """
        if self.embedder is None:
            return None
        try:
            result = await self.embedder.embed_documents(texts)
            return np.array(result)
        except Exception:
            return None

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return dot / max(norm, 1e-8)


# ──────────────────────────────────────────────────────────────────
# Cross-Encoder Reranker
# ──────────────────────────────────────────────────────────────────


class CrossEncoderReranker(Reranker):
    """Cross-encoder reranker using HuggingFace models.

    Uses a cross-encoder model (e.g., ms-marco-MiniLM) to score
    query-document pairs for highly accurate re-ranking.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 10,
        batch_size: int = 32,
        device: str = "cpu",
        **kwargs: Any,
    ):
        """Initialize the cross-encoder reranker.

        Args:
            model_name: HuggingFace cross-encoder model.
            top_k: Number of results.
            batch_size: Batch size for scoring.
            device: Device for inference.
            **kwargs: Additional arguments.
        """
        super().__init__(top_k=top_k, strategy=RerankStrategy.CROSS_ENCODER)
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None

    def _load_model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name, device=self.device)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for cross-encoder reranking. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> RerankResponse:
        """Re-rank using cross-encoder.

        Args:
            query: Search query.
            documents: Document list.
            top_k: Number of results.

        Returns:
            RerankResponse.
        """
        start = time.monotonic()
        top_k = top_k or self.top_k

        if not documents:
            return RerankResponse(
                query=query,
                results=[],
                strategy=self.strategy,
                input_count=0,
                output_count=0,
            )

        model = self._load_model()

        # Create query-document pairs
        pairs = [(query, d.get("text", "")[:1000]) for d in documents]

        # Score in a thread pool (CrossEncoder is synchronous)
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            ),
        )

        if hasattr(scores, 'tolist'):
            scores = scores.tolist()
        if not isinstance(scores, list):
            scores = [scores]

        # Sort by cross-encoder scores
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)

        results = []
        for new_rank, (orig_idx, score) in enumerate(indexed[:top_k]):
            doc = documents[orig_idx]
            results.append(RerankResult(
                chunk_id=doc.get("chunk_id", f"doc_{orig_idx}"),
                text=doc.get("text", "")[:500],
                score=score,
                original_rank=orig_idx,
                new_rank=new_rank,
                original_score=doc.get("score", 0.0),
                rerank_score=score,
                metadata=doc.get("metadata", {}),
            ))

        latency = (time.monotonic() - start) * 1000

        return RerankResponse(
            query=query,
            results=results,
            strategy=self.strategy,
            latency_ms=latency,
            input_count=len(documents),
            output_count=len(results),
        )


# ──────────────────────────────────────────────────────────────────
# LLM Judge Reranker
# ──────────────────────────────────────────────────────────────────


class LLMJudgeReranker(Reranker):
    """LLM-based reranker that uses an LLM to judge relevance.

    Sends query-document pairs to an LLM and asks it to rate relevance,
    providing high-quality but slower re-ranking.
    """

    def __init__(
        self,
        llm_complete_fn: Optional[Callable] = None,
        top_k: int = 10,
        max_documents: int = 20,
        **kwargs: Any,
    ):
        """Initialize the LLM judge reranker.

        Args:
            llm_complete_fn: Async function to call LLM.
            top_k: Number of results.
            max_documents: Max documents to evaluate (cost control).
            **kwargs: Additional arguments.
        """
        super().__init__(top_k=top_k, strategy=RerankStrategy.LLM_JUDGE)
        self.llm_complete_fn = llm_complete_fn
        self.max_documents = max_documents

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> RerankResponse:
        """Re-rank using LLM judge.

        Args:
            query: Search query.
            documents: Document list.
            top_k: Number of results.

        Returns:
            RerankResponse.
        """
        start = time.monotonic()
        top_k = top_k or self.top_k

        if not documents:
            return RerankResponse(
                query=query,
                results=[],
                strategy=self.strategy,
                input_count=0,
                output_count=0,
            )

        # Limit documents for cost
        docs_to_judge = documents[:self.max_documents]

        if self.llm_complete_fn is None:
            # Fallback to score-based reranking
            fallback = ScoreReranker(top_k=top_k)
            response = await fallback.rerank(query, documents, top_k)
            response.strategy = RerankStrategy.LLM_JUDGE
            return response

        # Judge each document
        scores = []
        for doc in docs_to_judge:
            score = await self._judge_document(query, doc)
            scores.append(score)

        # Sort by judged scores
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)

        results = []
        for new_rank, (orig_idx, score) in enumerate(indexed[:top_k]):
            doc = docs_to_judge[orig_idx]
            results.append(RerankResult(
                chunk_id=doc.get("chunk_id", f"doc_{orig_idx}"),
                text=doc.get("text", "")[:500],
                score=score,
                original_rank=orig_idx,
                new_rank=new_rank,
                original_score=doc.get("score", 0.0),
                rerank_score=score,
                metadata=doc.get("metadata", {}),
            ))

        latency = (time.monotonic() - start) * 1000

        return RerankResponse(
            query=query,
            results=results,
            strategy=self.strategy,
            latency_ms=latency,
            input_count=len(documents),
            output_count=len(results),
        )

    async def _judge_document(self, query: str, doc: Dict[str, Any]) -> float:
        """Have an LLM judge document relevance.

        Args:
            query: Search query.
            doc: Document dict.

        Returns:
            Relevance score (0-10).
        """
        if self.llm_complete_fn is None:
            return doc.get("score", 0.0)

        text = doc.get("text", "")[:1000]
        prompt = (
            f"Rate the relevance of the following document to the query "
            f"on a scale of 0 to 10, where 0 is completely irrelevant "
            f"and 10 is perfectly relevant.\n\n"
            f"Query: {query}\n\n"
            f"Document: {text}\n\n"
            f"Relevance score (0-10):"
        )

        try:
            response = await self.llm_complete_fn(prompt)
            # Extract numeric score
            import re
            match = re.search(r'(\d+(?:\.\d+)?)', str(response))
            if match:
                score = float(match.group(1))
                return min(10.0, max(0.0, score)) / 10.0
        except Exception:
            pass

        return doc.get("score", 0.0)


# ──────────────────────────────────────────────────────────────────
# Multi-Stage Reranking Pipeline
# ──────────────────────────────────────────────────────────────────


class RerankingPipeline:
    """Multi-stage re-ranking pipeline.

    Chain multiple rerankers for progressive refinement:
    Stage 1: Fast score-based filter (100 → 50)
    Stage 2: MMR diversity (50 → 20)
    Stage 3: Cross-encoder precision (20 → 10)
    """

    def __init__(self, rerankers: Optional[List[Reranker]] = None):
        """Initialize the pipeline.

        Args:
            rerankers: Ordered list of rerankers.
        """
        self.rerankers = rerankers or []

    def add_stage(self, reranker: Reranker) -> None:
        """Add a re-ranking stage.

        Args:
            reranker: Reranker instance.
        """
        self.rerankers.append(reranker)

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        final_top_k: int = 10,
    ) -> RerankResponse:
        """Execute the multi-stage pipeline.

        Args:
            query: Search query.
            documents: Initial document list.
            final_top_k: Final number of results.

        Returns:
            RerankResponse.
        """
        current_docs = list(documents)
        total_latency = 0.0

        for i, reranker in enumerate(self.rerankers):
            is_last = (i == len(self.rerankers) - 1)
            stage_top_k = final_top_k if is_last else reranker.top_k

            response = await reranker.rerank(query, current_docs, stage_top_k)
            total_latency += response.latency_ms

            # Convert back to dict format for next stage
            current_docs = [
                {
                    "chunk_id": r.chunk_id,
                    "text": r.text,
                    "score": r.rerank_score,
                    "metadata": r.metadata,
                }
                for r in response.results
            ]

        # Return final stage results
        if self.rerankers:
            final_response = await self.rerankers[-1].rerank(
                query, current_docs, final_top_k
            )
            final_response.latency_ms = total_latency
            return final_response

        # No stages — return as-is
        return RerankResponse(
            query=query,
            results=[],
            strategy=RerankStrategy.SCORE_COMBINATION,
            input_count=len(documents),
            output_count=0,
        )


# ──────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────


def create_reranker(
    strategy: RerankStrategy = RerankStrategy.SCORE_COMBINATION,
    **kwargs: Any,
) -> Reranker:
    """Create a reranker instance.

    Args:
        strategy: Reranking strategy.
        **kwargs: Strategy-specific arguments.

    Returns:
        A Reranker instance.
    """
    if strategy == RerankStrategy.SCORE_COMBINATION:
        return ScoreReranker(**kwargs)
    elif strategy == RerankStrategy.MMR:
        return MMRReranker(**kwargs)
    elif strategy == RerankStrategy.CROSS_ENCODER:
        return CrossEncoderReranker(**kwargs)
    elif strategy == RerankStrategy.LLM_JUDGE:
        return LLMJudgeReranker(**kwargs)
    else:
        raise ValueError(
            f"Unknown rerank strategy: {strategy}. "
            f"Available: {[s.value for s in RerankStrategy]}"
        )
