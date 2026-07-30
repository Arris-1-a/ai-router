"""
Response caching layer with multiple cache backends.

Supports:
  - LRU in-memory cache for fast lookups
  - Semantic similarity cache for fuzzy deduplication
  - TTL-based expiration
  - Cache key generation with configurable strategies
"""

from __future__ import annotations

import hashlib
import heapq
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np


class CacheBackend(Protocol):
    """Protocol for cache backend implementations."""

    def get(self, key: str) -> Optional[Any]: ...
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None: ...
    def delete(self, key: str) -> bool: ...
    def clear(self) -> None: ...
    def stats(self) -> Dict[str, Any]: ...


# ──────────────────────────────────────────────────────────────────
# Cache Entry
# ──────────────────────────────────────────────────────────────────


@dataclass
class CacheEntry:
    """A single entry in the cache."""

    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    ttl: Optional[float] = None  # seconds
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self, now: Optional[float] = None) -> bool:
        """Check if this entry has expired.

        Args:
            now: Current timestamp (default: time.time()).

        Returns:
            True if the entry is expired.
        """
        if self.ttl is None:
            return False
        now = now or time.time()
        return (now - self.created_at) > self.ttl

    def touch(self) -> None:
        """Update access time and count."""
        self.last_accessed = time.time()
        self.access_count += 1


# ──────────────────────────────────────────────────────────────────
# LRU Cache
# ──────────────────────────────────────────────────────────────────


class LRUCache:
    """Thread-safe LRU (Least Recently Used) cache with TTL support.

    Evicts the least recently accessed entries when the capacity is exceeded.
    """

    def __init__(
        self,
        max_size: int = 1000,
        max_memory_bytes: Optional[int] = None,
        default_ttl: Optional[float] = 3600.0,
    ):
        """Initialize the LRU cache.

        Args:
            max_size: Maximum number of entries.
            max_memory_bytes: Maximum memory usage in bytes.
            default_ttl: Default TTL for new entries in seconds.
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_bytes
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found/expired.
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None

            entry.touch()
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a value in the cache.

        Args:
            key: Cache key.
            value: Value to cache.
            ttl: TTL in seconds (defaults to self.default_ttl).
            metadata: Optional metadata dict.
        """
        import sys

        with self._lock:
            ttl = ttl if ttl is not None else self.default_ttl

            # Estimate size
            try:
                size_bytes = sys.getsizeof(value)
            except (TypeError, AttributeError):
                size_bytes = 1024

            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl,
                size_bytes=size_bytes,
                metadata=metadata or {},
            )

            # If key exists, update in place
            if key in self._cache:
                old = self._cache[key]
                self._cache[key] = entry
                self._cache.move_to_end(key)
                return

            # Check capacity before adding
            while len(self._cache) >= self.max_size:
                self._evict_one()

            # Check memory constraint
            if self.max_memory_bytes is not None:
                while self._estimate_memory() + size_bytes > self.max_memory_bytes:
                    if not self._evict_one():
                        break

            self._cache[key] = entry

    def delete(self, key: str) -> bool:
        """Delete a cache entry.

        Args:
            key: Cache key.

        Returns:
            True if the entry was found and deleted.
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def contains(self, key: str) -> bool:
        """Check if key exists and is not expired.

        Args:
            key: Cache key.

        Returns:
            True if the key exists and is valid.
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired():
                del self._cache[key]
                return False
            return True

    def size(self) -> int:
        """Get current number of entries."""
        with self._lock:
            return len(self._cache)

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with hit rate, miss rate, evictions, size.
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 4),
                "evictions": self._evictions,
                "estimated_memory_bytes": self._estimate_memory(),
            }

    def _evict_one(self) -> bool:
        """Evict the least recently used entry.

        Returns:
            True if an entry was evicted.
        """
        if not self._cache:
            return False
        self._cache.popitem(last=False)
        self._evictions += 1
        return True

    def _estimate_memory(self) -> int:
        """Estimate total memory usage."""
        return sum(e.size_bytes for e in self._cache.values())

    def keys(self) -> List[str]:
        """Get all valid (non-expired) keys."""
        with self._lock:
            now = time.time()
            return [k for k, e in self._cache.items() if not e.is_expired(now)]


# ──────────────────────────────────────────────────────────────────
# Semantic Similarity Cache
# ──────────────────────────────────────────────────────────────────


class SemanticCache:
    """Cache that finds similar entries using semantic similarity.

    When a cache miss occurs, checks if a semantically similar query
    was previously cached, avoiding duplicate API calls for
    near-identical requests.
    """

    def __init__(
        self,
        lru_cache: Optional[LRUCache] = None,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        similarity_threshold: float = 0.92,
        max_similar_candidates: int = 20,
    ):
        """Initialize the semantic cache.

        Args:
            lru_cache: Underlying LRU cache for storage.
            embed_fn: Function to generate text embeddings.
            similarity_threshold: Minimum cosine similarity for a cache hit.
            max_similar_candidates: Max entries to compare for similarity.
        """
        self.cache = lru_cache or LRUCache(max_size=5000)
        self.embed_fn = embed_fn or self._default_embed_fn
        self.similarity_threshold = similarity_threshold
        self.max_similar_candidates = max_similar_candidates
        self._embedding_store: Dict[str, List[float]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _default_embed_fn(text: str) -> List[float]:
        """Default embedding function using TF-based hashing."""
        n = 3
        dim = 256
        vec = [0.0] * dim
        text_lower = text.lower()
        for i in range(len(text_lower) - n + 1):
            ngram = text_lower[i : i + n]
            h = int(hashlib.md5(ngram.encode()).hexdigest(), 16) % dim
            vec[h] += 1.0
        norm = np.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def get_exact(self, key: str) -> Optional[Any]:
        """Get exact cache match.

        Args:
            key: Exact cache key.

        Returns:
            Cached value or None.
        """
        return self.cache.get(key)

    def get_similar(self, query_text: str) -> Optional[Tuple[str, Any, float]]:
        """Find the most semantically similar cached entry.

        Args:
            query_text: The query text to find similar matches for.

        Returns:
            Tuple of (matched_key, value, similarity) or None.
        """
        query_embedding = self.embed_fn(query_text)

        with self._lock:
            # Get candidate keys
            all_keys = self.cache.keys()
            candidates = all_keys[-self.max_similar_candidates :]

            best_key = None
            best_sim = -1.0

            for key in candidates:
                stored_emb = self._embedding_store.get(key)
                if stored_emb is None:
                    continue
                sim = self._cosine_similarity(query_embedding, stored_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_key = key

            if best_key is not None and best_sim >= self.similarity_threshold:
                value = self.cache.get(best_key)
                if value is not None:
                    return (best_key, value, best_sim)

        return None

    def set(
        self,
        key: str,
        value: Any,
        query_text: Optional[str] = None,
        ttl: Optional[float] = None,
    ) -> None:
        """Store a value with its embedding for semantic lookup.

        Args:
            key: Cache key.
            value: Value to cache.
            query_text: Text to generate embedding from (for semantic matching).
            ttl: TTL in seconds.
        """
        self.cache.set(key, value, ttl=ttl)
        if query_text:
            text = query_text
        else:
            text = key
        with self._lock:
            self._embedding_store[key] = self.embed_fn(text)

    def delete(self, key: str) -> bool:
        """Delete a cache entry and its embedding.

        Args:
            key: Cache key.

        Returns:
            True if deleted.
        """
        with self._lock:
            self._embedding_store.pop(key, None)
        return self.cache.delete(key)

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._embedding_store.clear()
        self.cache.clear()

    def stats(self) -> Dict[str, Any]:
        """Get combined statistics."""
        base_stats = self.cache.stats()
        base_stats["semantic_entries"] = len(self._embedding_store)
        return base_stats

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = np.sqrt(sum(x * x for x in a))
        norm_b = np.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


# ──────────────────────────────────────────────────────────────────
# Cache Manager
# ──────────────────────────────────────────────────────────────────


class CacheKeyStrategy(str, Enum):
    """Strategies for generating cache keys."""

    EXACT = "exact"         # Hash of exact messages
    NORMALIZED = "normalized"  # Normalize whitespace/case before hashing
    STRUCTURE = "structure"  # Only hash message structure (roles, not content)
    CUSTOM = "custom"       # Use a custom key function


class CacheManager:
    """High-level cache manager for LLM responses.

    Combines LRU + semantic caching with multiple key strategies.
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: float = 3600.0,
        key_strategy: CacheKeyStrategy = CacheKeyStrategy.NORMALIZED,
        enable_semantic: bool = True,
        similarity_threshold: float = 0.92,
        custom_key_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
    ):
        """Initialize the cache manager.

        Args:
            max_size: Max LRU cache entries.
            default_ttl: Default TTL in seconds.
            key_strategy: How to generate cache keys.
            enable_semantic: Enable semantic similarity matching.
            similarity_threshold: Threshold for semantic match.
            custom_key_fn: Custom key generation function.
        """
        self.lru_cache = LRUCache(max_size=max_size, default_ttl=default_ttl)
        self.semantic_cache: Optional[SemanticCache] = None
        if enable_semantic:
            self.semantic_cache = SemanticCache(
                lru_cache=self.lru_cache,
                similarity_threshold=similarity_threshold,
            )
        self.key_strategy = key_strategy
        self.custom_key_fn = custom_key_fn

    def generate_key(self, request_data: Dict[str, Any]) -> str:
        """Generate a cache key from request data.

        Args:
            request_data: Dictionary with messages, model, params.

        Returns:
            Cache key string.
        """
        if self.key_strategy == CacheKeyStrategy.CUSTOM and self.custom_key_fn:
            return self.custom_key_fn(request_data)

        if self.key_strategy == CacheKeyStrategy.STRUCTURE:
            # Only hash structure: roles + model
            messages = request_data.get("messages", [])
            structure = {
                "model": request_data.get("model", ""),
                "roles": [m.get("role", "") for m in messages],
                "num_messages": len(messages),
            }
            return self._hash_dict(structure)

        if self.key_strategy == CacheKeyStrategy.NORMALIZED:
            # Normalize whitespace and case
            normalized = self._normalize_request(request_data)
            return self._hash_dict(normalized)

        # EXACT strategy
        key_data = {
            "model": request_data.get("model", ""),
            "messages": request_data.get("messages", []),
            "temperature": request_data.get("temperature", 0.7),
            "max_tokens": request_data.get("max_tokens", 1024),
        }
        return self._hash_dict(key_data)

    def get(
        self,
        request_data: Dict[str, Any],
        query_text: Optional[str] = None,
    ) -> Optional[Any]:
        """Get a cached response.

        First tries exact key match, then semantic match if enabled.

        Args:
            request_data: Request data dict.
            query_text: Text for semantic matching.

        Returns:
            Cached value or None.
        """
        # Exact match
        key = self.generate_key(request_data)
        value = self.lru_cache.get(key)
        if value is not None:
            return value

        # Semantic match
        if self.semantic_cache is not None:
            search_text = query_text or self._extract_text(request_data)
            result = self.semantic_cache.get_similar(search_text)
            if result is not None:
                return result[1]

        return None

    def set(
        self,
        request_data: Dict[str, Any],
        response: Any,
        ttl: Optional[float] = None,
        query_text: Optional[str] = None,
    ) -> None:
        """Cache a response.

        Args:
            request_data: Request data for key generation.
            response: Response to cache.
            ttl: TTL override.
            query_text: Text for semantic cache.
        """
        key = self.generate_key(request_data)
        if self.semantic_cache is not None:
            self.semantic_cache.set(
                key=key,
                value=response,
                query_text=query_text or self._extract_text(request_data),
                ttl=ttl,
            )
        else:
            self.lru_cache.set(key, response, ttl=ttl)

    def delete(self, request_data: Dict[str, Any]) -> bool:
        """Delete a cached entry.

        Args:
            request_data: Request data to generate key from.

        Returns:
            True if deleted.
        """
        key = self.generate_key(request_data)
        if self.semantic_cache:
            return self.semantic_cache.delete(key)
        return self.lru_cache.delete(key)

    def clear(self) -> None:
        """Clear all cached entries."""
        if self.semantic_cache:
            self.semantic_cache.clear()
        self.lru_cache.clear()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self.semantic_cache:
            return self.semantic_cache.stats()
        return self.lru_cache.stats()

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _hash_dict(data: Dict[str, Any]) -> str:
        """Hash a dictionary to a cache key."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    @staticmethod
    def _normalize_request(data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize request for consistent hashing."""
        normalized = {
            "model": data.get("model", ""),
            "temperature": data.get("temperature", 0.7),
            "max_tokens": data.get("max_tokens", 1024),
        }
        messages = data.get("messages", [])
        norm_msgs = []
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, str):
                content = " ".join(content.lower().split())
            norm_msgs.append({
                "role": m.get("role", "user"),
                "content": content,
            })
        normalized["messages"] = norm_msgs
        return normalized

    @staticmethod
    def _extract_text(request_data: Dict[str, Any]) -> str:
        """Extract concatenated text from request messages."""
        messages = request_data.get("messages", [])
        parts = []
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        parts.append(item.get("text", ""))
        return " ".join(parts)
