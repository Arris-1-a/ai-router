"""
Multi-backend embedding generation for RAG pipelines.

Supports:
  - OpenAI embeddings (text-embedding-3-small/large, ada-002)
  - Sentence-transformers (local models via HuggingFace)
  - Caching to avoid redundant API calls
  - Batch processing for efficiency
  - Dimension normalization
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class EmbeddingBackend(str, Enum):
    """Available embedding backends."""

    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    HUGGINGFACE_API = "huggingface_api"
    CUSTOM = "custom"


@dataclass
class EmbeddingResult:
    """Result of an embedding operation."""

    embeddings: List[List[float]]
    model: str
    dimensions: int
    latency_ms: float = 0.0
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────
# Base Embedder
# ──────────────────────────────────────────────────────────────────


class Embedder(ABC):
    """Abstract base class for embedding generators."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        batch_size: int = 32,
        normalize: bool = True,
        cache_size: int = 10000,
    ):
        """Initialize the embedder.

        Args:
            model: Model name or path.
            batch_size: Batch size for processing.
            normalize: Whether to L2-normalize embeddings.
            cache_size: Size of the embedding cache.
        """
        self.model = model
        self.batch_size = batch_size
        self.normalize = normalize
        self._cache: Dict[str, List[float]] = {}
        self._cache_size = cache_size
        self._total_requests = 0
        self._cache_hits = 0

    @abstractmethod
    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings.

        Returns:
            List of embedding vectors.
        """
        ...

    async def embed(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False,
    ) -> EmbeddingResult:
        """Generate embeddings for one or more texts.

        Args:
            texts: Single text or list of texts.
            show_progress: Whether to show progress indicator.

        Returns:
            EmbeddingResult with embeddings and metadata.
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return EmbeddingResult(
                embeddings=[],
                model=self.model,
                dimensions=0,
            )

        start_time = time.monotonic()

        # Check cache first
        uncached_texts = []
        uncached_indices = []
        cached_embeddings: Dict[int, List[float]] = {}

        for i, text in enumerate(texts):
            cache_key = self._cache_key(text)
            if cache_key in self._cache:
                cached_embeddings[i] = self._cache[cache_key]
                self._cache_hits += 1
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        self._total_requests += len(uncached_texts)

        # Generate embeddings for uncached texts
        new_embeddings = []
        if uncached_texts:
            for batch_start in range(0, len(uncached_texts), self.batch_size):
                batch = uncached_texts[batch_start : batch_start + self.batch_size]
                batch_embeddings = await self._embed_batch(batch)

                if self.normalize:
                    batch_embeddings = [self._l2_normalize(emb) for emb in batch_embeddings]

                new_embeddings.extend(batch_embeddings)

                # Cache results
                for j, emb in enumerate(batch_embeddings):
                    text = batch[j]
                    cache_key = self._cache_key(text)
                    self._add_to_cache(cache_key, emb)

        # Merge cached and new embeddings
        all_embeddings: List[Optional[List[float]]] = [None] * len(texts)
        for idx, emb in cached_embeddings.items():
            all_embeddings[idx] = emb
        for j, idx in enumerate(uncached_indices):
            all_embeddings[idx] = new_embeddings[j]

        # All should be filled
        result_embeddings = [emb for emb in all_embeddings if emb is not None]

        latency = (time.monotonic() - start_time) * 1000

        return EmbeddingResult(
            embeddings=result_embeddings,
            model=self.model,
            dimensions=len(result_embeddings[0]) if result_embeddings else 0,
            latency_ms=latency,
            token_count=sum(len(t.split()) for t in texts),
            metadata={
                "cached": len(cached_embeddings),
                "uncached": len(uncached_texts),
                "total": len(texts),
            },
        )

    async def embed_query(self, query: str) -> List[float]:
        """Convenience method for single query embedding.

        Args:
            query: Query text.

        Returns:
            Embedding vector.
        """
        result = await self.embed(query)
        return result.embeddings[0] if result.embeddings else []

    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Convenience method for document embeddings.

        Args:
            documents: List of document texts.

        Returns:
            List of embedding vectors.
        """
        result = await self.embed(documents)
        return result.embeddings

    def similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Cosine similarity.
        """
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = np.sqrt(sum(x * x for x in a))
        norm_b = np.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def batch_similarity(
        self, query: List[float], documents: List[List[float]]
    ) -> List[float]:
        """Compute similarities between a query and multiple documents.

        Args:
            query: Query vector.
            documents: List of document vectors.

        Returns:
            List of similarity scores.
        """
        query_np = np.array(query)
        docs_np = np.array(documents)
        dots = np.dot(docs_np, query_np)
        norms = np.linalg.norm(docs_np, axis=1) * np.linalg.norm(query_np)
        norms = np.where(norms == 0, 1, norms)
        return (dots / norms).tolist()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache hit rate, size, etc.
        """
        return {
            "cache_size": len(self._cache),
            "max_size": self._cache_size,
            "total_requests": self._total_requests,
            "cache_hits": self._cache_hits,
            "hit_rate": (
                self._cache_hits / self._total_requests
                if self._total_requests > 0
                else 0.0
            ),
        }

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._total_requests = 0

    def _cache_key(self, text: str) -> str:
        """Generate cache key for text.

        Args:
            text: Input text.

        Returns:
            Cache key.
        """
        return hashlib.md5(f"{self.model}:{text}".encode()).hexdigest()

    def _add_to_cache(self, key: str, embedding: List[float]) -> None:
        """Add embedding to cache with size limit.

        Args:
            key: Cache key.
            embedding: Embedding vector.
        """
        if len(self._cache) >= self._cache_size:
            # Simple FIFO eviction
            first_key = next(iter(self._cache))
            del self._cache[first_key]
        self._cache[key] = embedding

    @staticmethod
    def _l2_normalize(vector: List[float]) -> List[float]:
        """L2-normalize a vector.

        Args:
            vector: Input vector.

        Returns:
            Normalized vector.
        """
        norm = np.sqrt(sum(x * x for x in vector))
        if norm == 0:
            return vector
        return [x / norm for x in vector]


# ──────────────────────────────────────────────────────────────────
# OpenAI Embedder
# ──────────────────────────────────────────────────────────────────


class OpenAIEmbedder(Embedder):
    """OpenAI embedding API wrapper."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        batch_size: int = 20,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize the OpenAI embedder.

        Args:
            api_key: OpenAI API key.
            model: Embedding model name.
            batch_size: Batch size for API calls.
            base_url: API base URL.
            **kwargs: Additional arguments.
        """
        import os
        super().__init__(model=model, batch_size=batch_size, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or "https://api.openai.com"

    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings via OpenAI API.

        Args:
            texts: List of texts.

        Returns:
            List of embedding vectors.
        """
        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(base_url=self.base_url, timeout=60.0) as client:
            response = await client.post(
                "/v1/embeddings",
                json={
                    "model": self.model,
                    "input": texts,
                },
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

            embeddings = [
                item["embedding"]
                for item in sorted(data["data"], key=lambda x: x["index"])
            ]
            return embeddings


# ──────────────────────────────────────────────────────────────────
# Sentence Transformers Embedder
# ──────────────────────────────────────────────────────────────────


class SentenceTransformersEmbedder(Embedder):
    """Local sentence-transformers embedder using HuggingFace models.

    Runs entirely locally with no API calls — best for privacy
    and offline usage. Supports any sentence-transformers model.
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
        **kwargs: Any,
    ):
        """Initialize the sentence-transformers embedder.

        Args:
            model: Sentence-transformers model name or path.
            device: Device to run on ('cpu', 'cuda', etc.).
            batch_size: Batch size for encoding.
            **kwargs: Additional arguments.
        """
        super().__init__(model=model, batch_size=batch_size, **kwargs)
        self.device = device
        self._model = None

    def _load_model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model, device=self.device)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for local embeddings. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings locally.

        Args:
            texts: List of texts.

        Returns:
            List of embedding vectors.
        """
        model = self._load_model()
        # Run in thread pool since sentence-transformers is synchronous
        loop = asyncio.get_event_loop()
        embeddings_array = await loop.run_in_executor(
            None,
            lambda: model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=self.normalize,
            ),
        )
        return embeddings_array.tolist()


# ──────────────────────────────────────────────────────────────────
# Embedder Factory
# ──────────────────────────────────────────────────────────────────


def create_embedder(
    backend: EmbeddingBackend = EmbeddingBackend.SENTENCE_TRANSFORMERS,
    model: Optional[str] = None,
    **kwargs: Any,
) -> Embedder:
    """Create an embedder instance.

    Args:
        backend: Embedding backend to use.
        model: Model name (uses backend default if None).
        **kwargs: Backend-specific arguments.

    Returns:
        An Embedder instance.

    Raises:
        ValueError: If backend is unknown.
    """
    if backend == EmbeddingBackend.OPENAI:
        return OpenAIEmbedder(
            model=model or "text-embedding-3-small",
            **kwargs,
        )
    elif backend == EmbeddingBackend.SENTENCE_TRANSFORMERS:
        return SentenceTransformersEmbedder(
            model=model or "all-MiniLM-L6-v2",
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown embedding backend: {backend}. "
            f"Available: {[e.value for e in EmbeddingBackend]}"
        )
