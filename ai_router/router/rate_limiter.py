"""
Token bucket rate limiter for LLM API requests.

Provides:
  - Token bucket algorithm for smooth rate limiting
  - Per-provider and global limits
  - Concurrent request limiting
  - Burst handling
  - Async and sync interfaces
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple


class LimitUnit(str, Enum):
    """Units for rate limits."""

    REQUESTS = "requests"
    TOKENS = "tokens"
    COST = "cost"


@dataclass
class RateLimit:
    """A rate limit configuration."""

    max_count: int              # Maximum units allowed
    per_seconds: float          # Time window in seconds
    unit: LimitUnit = LimitUnit.REQUESTS
    burst_multiplier: float = 1.5  # Allow burst above steady rate
    name: str = "default"


@dataclass
class RateLimitStatus:
    """Current status of a rate limiter."""

    limit: RateLimit
    current_tokens: float
    max_tokens: float
    is_limited: bool
    retry_after: float = 0.0  # Seconds until next token available
    requests_denied: int = 0
    requests_allowed: int = 0


@dataclass
class ConcurrencyLimit:
    """Limit on concurrent requests."""

    max_concurrent: int
    name: str = "default"


# ──────────────────────────────────────────────────────────────────
# Token Bucket
# ──────────────────────────────────────────────────────────────────


class TokenBucket:
    """Thread-safe token bucket implementation.

    Tokens are added at a steady rate (refill rate). Each request
    consumes tokens. If not enough tokens, the request is denied
    (or waits). Supports burst via higher initial token count.
    """

    def __init__(
        self,
        rate: float,           # Tokens per second
        capacity: float,       # Max tokens (bucket size)
        initial_tokens: Optional[float] = None,
        name: str = "default",
    ):
        """Initialize the token bucket.

        Args:
            rate: Token refill rate per second.
            capacity: Maximum token capacity.
            initial_tokens: Starting token count (default: capacity).
            name: Bucket name for identification.
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = initial_tokens if initial_tokens is not None else capacity
        self.last_refill = time.monotonic()
        self.name = name
        self._lock = threading.RLock()

        # Stats
        self.total_consumed = 0
        self.total_denied = 0
        self.total_waited = 0.0

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now

    def consume(self, tokens: float = 1.0) -> bool:
        """Try to consume tokens.

        Args:
            tokens: Number of tokens to consume.

        Returns:
            True if tokens were consumed, False if insufficient.
        """
        with self._lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                self.total_consumed += tokens
                return True

            self.total_denied += 1
            return False

    def try_consume(self, tokens: float = 1.0) -> Tuple[bool, float]:
        """Try to consume and return wait time if denied.

        Args:
            tokens: Number of tokens to consume.

        Returns:
            Tuple of (success, wait_time_in_seconds).
        """
        with self._lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                self.total_consumed += tokens
                return True, 0.0

            # Calculate how long to wait
            needed = tokens - self.tokens
            wait_time = needed / self.rate if self.rate > 0 else float("inf")
            self.total_denied += 1
            return False, wait_time

    def wait_and_consume(
        self,
        tokens: float = 1.0,
        timeout: Optional[float] = None,
    ) -> bool:
        """Block until tokens are available, then consume.

        Args:
            tokens: Number of tokens to consume.
            timeout: Max time to wait in seconds.

        Returns:
            True if tokens were consumed, False if timed out.
        """
        start = time.monotonic()

        while True:
            success, wait_time = self.try_consume(tokens)
            if success:
                waited = time.monotonic() - start
                self.total_waited += waited
                return True

            if timeout is not None:
                elapsed = time.monotonic() - start
                if elapsed + wait_time > timeout:
                    return False

            time.sleep(min(wait_time, 1.0))  # Cap sleep at 1s

    @property
    def available_tokens(self) -> float:
        """Get current available tokens."""
        with self._lock:
            self._refill()
            return self.tokens

    @property
    def retry_after(self) -> float:
        """Get seconds until next token is available."""
        with self._lock:
            self._refill()
            if self.tokens >= 1.0:
                return 0.0
            return (1.0 - self.tokens) / self.rate if self.rate > 0 else float("inf")

    def reset(self) -> None:
        """Reset bucket to full capacity."""
        with self._lock:
            self.tokens = self.capacity
            self.last_refill = time.monotonic()
            self.total_consumed = 0
            self.total_denied = 0
            self.total_waited = 0.0


# ──────────────────────────────────────────────────────────────────
# Async Token Bucket
# ──────────────────────────────────────────────────────────────────


class AsyncTokenBucket:
    """Async token bucket using asyncio for non-blocking rate limiting."""

    def __init__(
        self,
        rate: float,
        capacity: float,
        initial_tokens: Optional[float] = None,
        name: str = "default",
    ):
        """Initialize the async token bucket.

        Args:
            rate: Token refill rate per second.
            capacity: Maximum token capacity.
            initial_tokens: Starting token count.
            name: Bucket name.
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = initial_tokens if initial_tokens is not None else capacity
        self.last_refill = time.monotonic()
        self.name = name
        self._lock = asyncio.Lock()
        self._waiters: asyncio.Queue = asyncio.Queue()

        # Stats
        self.total_consumed = 0
        self.total_denied = 0

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now

    async def consume(self, tokens: float = 1.0) -> bool:
        """Try to consume tokens asynchronously.

        Args:
            tokens: Number of tokens to consume.

        Returns:
            True if consumed, False if denied.
        """
        async with self._lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                self.total_consumed += tokens
                return True
            self.total_denied += 1
            return False

    async def wait_and_consume(
        self,
        tokens: float = 1.0,
        timeout: Optional[float] = None,
    ) -> bool:
        """Wait asynchronously until tokens are available.

        Args:
            tokens: Number of tokens to consume.
            timeout: Max wait time.

        Returns:
            True if consumed, False if timed out.
        """
        start = time.monotonic()

        while True:
            async with self._lock:
                self._refill()
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    self.total_consumed += tokens
                    return True

                needed = tokens - self.tokens
                wait_time = needed / self.rate if self.rate > 0 else 1.0

            if timeout is not None:
                elapsed = time.monotonic() - start
                if elapsed >= timeout:
                    self.total_denied += 1
                    return False
                wait_time = min(wait_time, timeout - elapsed)

            await asyncio.sleep(min(wait_time, 1.0))

    @property
    def available_tokens(self) -> float:
        """Get available tokens (synchronous)."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        return min(self.capacity, self.tokens + elapsed * self.rate)


# ──────────────────────────────────────────────────────────────────
# Rate Limiter Manager
# ──────────────────────────────────────────────────────────────────


class TokenBucketRateLimiter:
    """Multi-bucket rate limiter for LLM API management.

    Manages rate limits per provider, per model, and globally.
    Supports both synchronous and asynchronous usage.
    """

    def __init__(self):
        """Initialize the rate limiter."""
        self._buckets: Dict[str, TokenBucket] = {}
        self._async_buckets: Dict[str, AsyncTokenBucket] = {}
        self._limits: Dict[str, RateLimit] = {}
        self._concurrency_limits: Dict[str, ConcurrencyLimit] = {}
        self._concurrent_count: Dict[str, int] = defaultdict(int)
        self._lock = threading.RLock()

    # ── Configuration ─────────────────────────────────────────────

    def add_limit(self, limit: RateLimit) -> None:
        """Add a rate limit.

        Args:
            limit: RateLimit configuration.
        """
        bucket_name = f"{limit.name}:{limit.unit.value}"
        rate = limit.max_count / limit.per_seconds
        capacity = limit.max_count * limit.burst_multiplier

        with self._lock:
            self._limits[bucket_name] = limit
            self._buckets[bucket_name] = TokenBucket(
                rate=rate,
                capacity=capacity,
                initial_tokens=capacity,
                name=bucket_name,
            )
            self._async_buckets[bucket_name] = AsyncTokenBucket(
                rate=rate,
                capacity=capacity,
                initial_tokens=capacity,
                name=bucket_name,
            )

    def set_provider_limit(
        self,
        provider: str,
        requests_per_minute: int = 60,
        tokens_per_minute: Optional[int] = None,
        burst_multiplier: float = 1.5,
    ) -> None:
        """Convenience method to set provider-specific limits.

        Args:
            provider: Provider name.
            requests_per_minute: Max RPM.
            tokens_per_minute: Max TPM (optional).
            burst_multiplier: Burst capacity multiplier.
        """
        self.add_limit(RateLimit(
            max_count=requests_per_minute,
            per_seconds=60.0,
            unit=LimitUnit.REQUESTS,
            burst_multiplier=burst_multiplier,
            name=f"provider:{provider}",
        ))

        if tokens_per_minute:
            self.add_limit(RateLimit(
                max_count=tokens_per_minute,
                per_seconds=60.0,
                unit=LimitUnit.TOKENS,
                burst_multiplier=burst_multiplier,
                name=f"provider:{provider}",
            ))

    def set_global_limit(
        self,
        requests_per_minute: int = 600,
        tokens_per_minute: Optional[int] = None,
    ) -> None:
        """Set global rate limits.

        Args:
            requests_per_minute: Max global RPM.
            tokens_per_minute: Max global TPM.
        """
        self.add_limit(RateLimit(
            max_count=requests_per_minute,
            per_seconds=60.0,
            unit=LimitUnit.REQUESTS,
            burst_multiplier=2.0,
            name="global",
        ))
        if tokens_per_minute:
            self.add_limit(RateLimit(
                max_count=tokens_per_minute,
                per_seconds=60.0,
                unit=LimitUnit.TOKENS,
                burst_multiplier=2.0,
                name="global",
            ))

    def set_concurrency_limit(self, name: str, max_concurrent: int) -> None:
        """Set a concurrency limit.

        Args:
            name: Limit name.
            max_concurrent: Max concurrent requests.
        """
        with self._lock:
            self._concurrency_limits[name] = ConcurrencyLimit(
                max_concurrent=max_concurrent,
                name=name,
            )

    # ── Synchronous API ───────────────────────────────────────────

    def acquire(self, provider: str, tokens: int = 1) -> bool:
        """Try to acquire permission for a request (non-blocking).

        Checks provider and global limits.

        Args:
            provider: Provider name.
            tokens: Estimated token count.

        Returns:
            True if allowed, False if rate limited.
        """
        # Check global limit
        global_req_key = "global:requests"
        if global_req_key in self._buckets:
            if not self._buckets[global_req_key].consume(1):
                return False

        # Check provider limit
        prov_req_key = f"provider:{provider}:requests"
        if prov_req_key in self._buckets:
            if not self._buckets[prov_req_key].consume(1):
                return False

        # Check token limits
        if tokens > 1:
            global_tok_key = "global:tokens"
            if global_tok_key in self._buckets:
                if not self._buckets[global_tok_key].consume(tokens):
                    # Return the request token
                    if global_req_key in self._buckets:
                        self._buckets[global_req_key].tokens += 1
                    return False

            prov_tok_key = f"provider:{provider}:tokens"
            if prov_tok_key in self._buckets:
                if not self._buckets[prov_tok_key].consume(tokens):
                    if global_req_key in self._buckets:
                        self._buckets[global_req_key].tokens += 1
                    return False

        return True

    def wait_and_acquire(
        self,
        provider: str,
        tokens: int = 1,
        timeout: Optional[float] = 30.0,
    ) -> bool:
        """Wait until a request is allowed (blocking).

        Args:
            provider: Provider name.
            tokens: Estimated token count.
            timeout: Max wait time.

        Returns:
            True if acquired, False if timed out.
        """
        relevant_buckets = []

        for name in ["global:requests", f"provider:{provider}:requests"]:
            if name in self._buckets:
                relevant_buckets.append((self._buckets[name], 1))

        for name in ["global:tokens", f"provider:{provider}:tokens"]:
            if name in self._buckets:
                relevant_buckets.append((self._buckets[name], tokens))

        if not relevant_buckets:
            return True

        start = time.monotonic()
        while True:
            all_ok = True
            for bucket, cost in relevant_buckets:
                if not bucket.consume(cost):
                    all_ok = False
                    # Return consumed tokens from already-checked buckets
                    for b2, c2 in relevant_buckets:
                        if b2 is bucket:
                            break
                        b2.tokens = min(b2.capacity, b2.tokens + c2)
                    break

            if all_ok:
                return True

            if timeout is not None and time.monotonic() - start >= timeout:
                return False

            time.sleep(0.1)

    # ── Async API ─────────────────────────────────────────────────

    async def async_acquire(
        self,
        provider: str,
        tokens: int = 1,
        timeout: Optional[float] = 30.0,
    ) -> bool:
        """Async version of acquire with wait.

        Args:
            provider: Provider name.
            tokens: Estimated token count.
            timeout: Max wait time.

        Returns:
            True if acquired, False if timed out.
        """
        relevant_buckets = []

        for name in ["global:requests", f"provider:{provider}:requests"]:
            if name in self._async_buckets:
                relevant_buckets.append((self._async_buckets[name], 1))

        for name in ["global:tokens", f"provider:{provider}:tokens"]:
            if name in self._async_buckets:
                relevant_buckets.append((self._async_buckets[name], tokens))

        if not relevant_buckets:
            return True

        start = time.monotonic()
        while True:
            all_ok = True
            consumed: list = []
            for bucket, cost in relevant_buckets:
                if not await bucket.consume(cost):
                    all_ok = False
                    # Roll back consumed tokens from already-checked buckets
                    for b2, c2 in consumed:
                        b2.tokens = min(b2.capacity, b2.tokens + c2)
                    break
                consumed.append((bucket, cost))

            if all_ok:
                return True

            if timeout is not None and time.monotonic() - start >= timeout:
                return False

            await asyncio.sleep(0.1)

    # ── Concurrency ───────────────────────────────────────────────

    def acquire_concurrent(self, name: str) -> bool:
        """Try to acquire a concurrency slot.

        Args:
            name: Concurrency limit name.

        Returns:
            True if slot acquired.
        """
        with self._lock:
            limit = self._concurrency_limits.get(name)
            if limit is None:
                return True
            if self._concurrent_count[name] < limit.max_concurrent:
                self._concurrent_count[name] += 1
                return True
            return False

    def release_concurrent(self, name: str) -> None:
        """Release a concurrency slot.

        Args:
            name: Concurrency limit name.
        """
        with self._lock:
            self._concurrent_count[name] = max(0, self._concurrent_count[name] - 1)

    # ── Status ────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Get status of all rate limit buckets.

        Returns:
            Dict with bucket statuses.
        """
        status = {}
        with self._lock:
            for name, bucket in self._buckets.items():
                limit = self._limits.get(name)
                status[name] = {
                    "available_tokens": round(bucket.available_tokens, 1),
                    "capacity": bucket.capacity,
                    "rate": bucket.rate,
                    "total_consumed": bucket.total_consumed,
                    "total_denied": bucket.total_denied,
                    "retry_after": round(bucket.retry_after, 3),
                }
                if limit:
                    status[name]["limit"] = {
                        "max": limit.max_count,
                        "per_seconds": limit.per_seconds,
                        "unit": limit.unit.value,
                    }

            # Concurrency status
            conc_status = {}
            for name, count in self._concurrent_count.items():
                limit = self._concurrency_limits.get(name)
                if limit:
                    conc_status[name] = {
                        "current": count,
                        "max": limit.max_concurrent,
                    }

            status["concurrency"] = conc_status

        return status

    def reset(self, name: Optional[str] = None) -> None:
        """Reset specific or all buckets.

        Args:
            name: Bucket name or None for all.
        """
        with self._lock:
            if name:
                if name in self._buckets:
                    self._buckets[name].reset()
                if name in self._async_buckets:
                    self._async_buckets[name] = AsyncTokenBucket(
                        rate=self._buckets[name].rate,
                        capacity=self._buckets[name].capacity,
                        name=name,
                    )
            else:
                for b in self._buckets.values():
                    b.reset()
                for name2, b2 in self._buckets.items():
                    self._async_buckets[name2] = AsyncTokenBucket(
                        rate=b2.rate,
                        capacity=b2.capacity,
                        name=name2,
                    )


# ──────────────────────────────────────────────────────────────────
# Decorator
# ──────────────────────────────────────────────────────────────────


def rate_limited(
    limiter: TokenBucketRateLimiter,
    provider: str,
    tokens: int = 1,
    timeout: float = 30.0,
):
    """Decorator to rate-limit async functions.

    Usage:
        @rate_limited(limiter, "openai")
        async def call_llm():
            ...
    """
    def decorator(func: Callable):
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not await limiter.async_acquire(provider, tokens, timeout):
                raise Exception(
                    f"Rate limit exceeded for {provider} "
                    f"(timeout={timeout}s)"
                )
            try:
                return await func(*args, **kwargs)
            finally:
                pass
        return wrapper
    return decorator
