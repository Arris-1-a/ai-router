"""Tests for cache, fallback, metrics, and rate limiter modules."""

import asyncio
import time
import pytest
from ai_router.router.cache import (
    CacheKeyStrategy,
    CacheManager,
    LRUCache,
    SemanticCache,
)
from ai_router.router.fallback import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    FallbackAction,
    FallbackChainEntry,
    FallbackManager,
    FallbackReason,
)
from ai_router.router.metrics import (
    Alert,
    AlertRule,
    MetricType,
    MetricsTracker,
    RequestMetrics,
)
from ai_router.router.rate_limiter import (
    TokenBucket,
    TokenBucketRateLimiter,
    RateLimit,
    LimitUnit,
)


# ── Cache Tests ───────────────────────────────────────────────────


class TestLRUCache:
    def test_set_and_get(self):
        cache = LRUCache(max_size=10)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_miss_returns_none(self):
        cache = LRUCache(max_size=10)
        assert cache.get("nonexistent") is None

    def test_expiry(self):
        cache = LRUCache(max_size=10, default_ttl=0.01)
        cache.set("key1", "value1")
        time.sleep(0.02)
        assert cache.get("key1") is None

    def test_eviction(self):
        cache = LRUCache(max_size=3)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.set("k3", "v3")
        cache.set("k4", "v4")  # Should evict k1
        assert cache.get("k1") is None
        assert cache.get("k4") == "v4"

    def test_lru_order(self):
        cache = LRUCache(max_size=3)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.set("k3", "v3")
        cache.get("k1")  # Makes k1 most recent
        cache.set("k4", "v4")
        assert cache.get("k1") == "v1"  # k1 survived
        assert cache.get("k2") is None  # k2 evicted

    def test_contains(self):
        cache = LRUCache(max_size=10)
        cache.set("key1", "value1")
        assert cache.contains("key1") is True
        assert cache.contains("key2") is False

    def test_delete(self):
        cache = LRUCache(max_size=10)
        cache.set("key1", "value1")
        assert cache.delete("key1") is True
        assert cache.delete("key1") is False

    def test_stats(self):
        cache = LRUCache(max_size=10)
        cache.set("k1", "v1")
        cache.get("k1")
        cache.get("k2")  # miss
        stats = cache.stats()
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_clear(self):
        cache = LRUCache(max_size=10)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.clear()
        assert cache.size() == 0


class TestSemanticCache:
    def test_exact_match(self):
        sc = SemanticCache()
        sc.set("k1", "value1", query_text="hello world")
        result = sc.get_exact("k1")
        assert result == "value1"

    def test_similar_match(self):
        sc = SemanticCache(similarity_threshold=0.5)
        sc.set("k1", "value1", query_text="hello world how are you")
        result = sc.get_similar("hello world how are you today")
        assert result is not None
        assert result[1] == "value1"

    def test_dissimilar_no_match(self):
        sc = SemanticCache(similarity_threshold=0.95)
        sc.set("k1", "value1", query_text="hello world")
        result = sc.get_similar("completely different topic about programming")
        assert result is None

    def test_delete_removes_embedding(self):
        sc = SemanticCache()
        sc.set("k1", "value1", query_text="hello")
        sc.delete("k1")
        assert sc.get_exact("k1") is None


class TestCacheManager:
    def test_key_generation_exact(self):
        cm = CacheManager(key_strategy=CacheKeyStrategy.EXACT)
        key1 = cm.generate_key({"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]})
        key2 = cm.generate_key({"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]})
        assert key1 == key2

    def test_key_generation_normalized(self):
        cm = CacheManager(key_strategy=CacheKeyStrategy.NORMALIZED)
        key1 = cm.generate_key({"model": "gpt-4", "messages": [{"role": "user", "content": "Hello   World"}]})
        key2 = cm.generate_key({"model": "gpt-4", "messages": [{"role": "user", "content": "hello world"}]})
        assert key1 == key2

    def test_set_and_get(self):
        cm = CacheManager()
        cm.set({"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]}, "response1")
        result = cm.get({"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]})
        assert result == "response1"

    def test_cache_miss(self):
        cm = CacheManager()
        result = cm.get({"model": "gpt-4", "messages": [{"role": "user", "content": "New"}]})
        assert result is None


# ── Fallback Tests ───────────────────────────────────────────────


class TestCircuitBreaker:
    def test_initial_state(self):
        cb = CircuitBreaker()
        assert cb.get_state("test") == CircuitState.CLOSED

    def test_opens_after_failures(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))
        for _ in range(3):
            cb.record_failure("test")
        assert cb.get_state("test") == CircuitState.OPEN

    def test_resets_after_successes(self):
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=3, success_threshold=2, timeout=0.01
        ))
        # Open the circuit
        for _ in range(3):
            cb.record_failure("test")
        assert cb.get_state("test") == CircuitState.OPEN

        # Wait for timeout
        time.sleep(0.02)
        # Check allowed triggers half-open
        assert cb.is_allowed("test") is True
        assert cb.get_state("test") == CircuitState.HALF_OPEN

        # Record successes
        cb.record_success("test")
        cb.record_success("test")
        assert cb.get_state("test") == CircuitState.CLOSED

    def test_reopens_in_half_open(self):
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=3, timeout=0.01
        ))
        for _ in range(3):
            cb.record_failure("test")
        time.sleep(0.02)
        cb.is_allowed("test")
        cb.record_failure("test")
        assert cb.get_state("test") == CircuitState.OPEN

    def test_reset(self):
        cb = CircuitBreaker()
        for _ in range(5):
            cb.record_failure("test")
        cb.reset("test")
        assert cb.get_state("test") == CircuitState.CLOSED


class TestFallbackManager:
    def test_add_fallback(self):
        fm = FallbackManager()
        fm.add_fallback("primary", "backup", "backup-model")
        targets = fm.get_fallback_targets("primary")
        assert len(targets) == 1
        assert targets[0].provider == "backup"

    def test_set_fallback_chain(self):
        fm = FallbackManager()
        chain = [
            FallbackChainEntry(provider="a", model="m1"),
            FallbackChainEntry(provider="b", model="m2"),
        ]
        fm.set_fallback_chain("primary", chain)
        targets = fm.get_fallback_targets("primary")
        assert len(targets) == 2

    def test_circuit_breaker_integration(self):
        fm = FallbackManager(CircuitBreakerConfig(failure_threshold=1))
        fm.circuit_breaker.record_failure("a:m1")
        assert fm.is_target_available("a:m1") is False

    def test_get_stats(self):
        fm = FallbackManager()
        fm.add_fallback("p", "fb", "model")
        stats = fm.get_stats()
        assert stats["total_fallbacks"] == 0
        assert "p" in stats["fallback_chains"]


# ── Metrics Tests ────────────────────────────────────────────────


class TestMetricsTracker:
    def test_start_and_complete(self):
        tracker = MetricsTracker()
        metrics = tracker.start_request("openai", "gpt-4o")
        assert metrics.provider == "openai"
        assert metrics.model == "gpt-4o"
        assert metrics.success is True

        tracker.complete_request(metrics, success=True, latency_ms=500, prompt_tokens=100,
                                 completion_tokens=50, cost=0.001)
        assert metrics.latency_ms == 500
        assert metrics.total_tokens == 150
        assert metrics.cost == 0.001

    def test_get_recent(self):
        tracker = MetricsTracker()
        for i in range(20):
            m = tracker.start_request("openai", "gpt-4o")
            tracker.complete_request(m, latency_ms=100)
        recent = tracker.get_recent_requests(limit=10)
        assert len(recent) == 10

    def test_global_stats(self):
        tracker = MetricsTracker()
        for i in range(10):
            m = tracker.start_request("openai", "gpt-4o")
            tracker.complete_request(m, success=(i < 8), cost=0.01)
        stats = tracker.get_global_stats()
        assert stats["total_requests"] == 10
        assert stats["total_successes"] == 8
        assert stats["total_errors"] == 2

    def test_alert_rule(self):
        tracker = MetricsTracker()
        tracker.add_alert_rule(AlertRule(
            name="high_latency",
            metric=MetricType.LATENCY,
            condition="gt",
            threshold=500,
            window_seconds=10,
            cooldown_seconds=0,
        ))

        # Should not trigger
        m1 = tracker.start_request("openai", "gpt-4o")
        tracker.complete_request(m1, latency_ms=100)

        # Should trigger
        m2 = tracker.start_request("openai", "gpt-4o")
        tracker.complete_request(m2, latency_ms=600)

        alerts = tracker.get_alerts()
        assert len(alerts) >= 1
        assert alerts[-1].rule_name == "high_latency"


# ── Rate Limiter Tests ────────────────────────────────────────────


class TestTokenBucket:
    def test_consume(self):
        bucket = TokenBucket(rate=10, capacity=10)
        assert bucket.consume(5) is True
        assert bucket.available_tokens < 10

    def test_consume_too_many(self):
        bucket = TokenBucket(rate=10, capacity=5)
        assert bucket.consume(10) is False

    def test_refill(self):
        bucket = TokenBucket(rate=100, capacity=10)
        bucket.consume(10)
        assert bucket.available_tokens < 10
        time.sleep(0.1)
        assert bucket.available_tokens > 0

    def test_retry_after(self):
        bucket = TokenBucket(rate=10, capacity=5)
        bucket.consume(5)  # Empty
        after = bucket.retry_after
        assert after > 0

    def test_reset(self):
        bucket = TokenBucket(rate=10, capacity=10)
        bucket.consume(10)
        bucket.reset()
        assert bucket.available_tokens == 10


class TestRateLimiter:
    def test_set_provider_limit(self):
        limiter = TokenBucketRateLimiter()
        limiter.set_provider_limit("openai", requests_per_minute=60)
        assert limiter.acquire("openai") is True

    def test_rate_limit_enforcement(self):
        limiter = TokenBucketRateLimiter()
        limiter.set_provider_limit("openai", requests_per_minute=2)
        # First 2 should pass (up to burst multiplier)
        results = [limiter.acquire("openai") for _ in range(5)]
        # At least some should fail
        assert any(r is False for r in results)

    def test_concurrency(self):
        limiter = TokenBucketRateLimiter()
        limiter.set_concurrency_limit("test", max_concurrent=2)
        assert limiter.acquire_concurrent("test") is True
        assert limiter.acquire_concurrent("test") is True
        assert limiter.acquire_concurrent("test") is False
        limiter.release_concurrent("test")
        assert limiter.acquire_concurrent("test") is True

    def test_global_limit(self):
        limiter = TokenBucketRateLimiter()
        limiter.set_global_limit(requests_per_minute=1)
        assert limiter.acquire("any_provider") is True
        assert limiter.acquire("any_provider") is False  # Exhausted burst too

    def test_get_status(self):
        limiter = TokenBucketRateLimiter()
        limiter.set_provider_limit("openai", requests_per_minute=10)
        status = limiter.get_status()
        assert "provider:openai:requests" in status
