"""
Fault-tolerant fallback and degradation management.

Handles provider failures gracefully by:
  - Automatic retry with exponential backoff
  - Multi-level fallback chains
  - Circuit breaker pattern for failing providers
  - Graceful degradation (e.g., simpler model on failure)
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)


# ──────────────────────────────────────────────────────────────────
# Data models
# ──────────────────────────────────────────────────────────────────


class FallbackReason(str, Enum):
    """Reasons for triggering a fallback."""

    PROVIDER_ERROR = "provider_error"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTH_ERROR = "auth_error"
    CONTEXT_OVERFLOW = "context_overflow"
    CIRCUIT_OPEN = "circuit_open"
    HEALTH_CHECK_FAIL = "health_check_fail"
    USER_DEFINED = "user_defined"


class CircuitState(str, Enum):
    """States for the circuit breaker pattern."""

    CLOSED = "closed"         # Normal operation
    OPEN = "open"             # Failing, reject requests
    HALF_OPEN = "half_open"   # Testing if recovered


@dataclass
class FallbackAction:
    """Describes a fallback action taken."""

    reason: FallbackReason
    from_target: str
    to_target: str
    attempt: int
    error_message: str = ""
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class CircuitBreakerConfig:
    """Configuration for the circuit breaker."""

    failure_threshold: int = 5         # Consecutive failures to open circuit
    success_threshold: int = 2         # Successes in half-open to close
    timeout: float = 60.0              # Seconds before trying half-open
    half_open_max_requests: int = 1    # Max requests in half-open state


@dataclass
class FallbackChainEntry:
    """A single entry in a fallback chain."""

    provider: str
    model: str
    weight: float = 1.0
    max_retries: int = 2


# ──────────────────────────────────────────────────────────────────
# Circuit Breaker
# ──────────────────────────────────────────────────────────────────


class CircuitBreaker:
    """Circuit breaker pattern implementation for LLM providers.

    Monitors failures and opens the circuit when a provider is unhealthy,
    preventing cascading failures and wasted API calls.
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """Initialize the circuit breaker.

        Args:
            config: Circuit breaker configuration.
        """
        self.config = config or CircuitBreakerConfig()
        self._state: Dict[str, CircuitState] = defaultdict(lambda: CircuitState.CLOSED)
        self._failure_count: Dict[str, int] = defaultdict(int)
        self._success_count: Dict[str, int] = defaultdict(int)
        self._last_failure_time: Dict[str, float] = {}
        self._opened_at: Dict[str, float] = {}
        self._total_failures: Dict[str, int] = defaultdict(int)
        self._total_successes: Dict[str, int] = defaultdict(int)

    def record_success(self, target_key: str) -> None:
        """Record a successful request.

        Args:
            target_key: Target identifier.
        """
        self._total_successes[target_key] += 1
        current = self._state[target_key]

        if current == CircuitState.HALF_OPEN:
            self._success_count[target_key] += 1
            if self._success_count[target_key] >= self.config.success_threshold:
                self._state[target_key] = CircuitState.CLOSED
                self._failure_count[target_key] = 0
                self._success_count[target_key] = 0

        elif current == CircuitState.CLOSED:
            self._failure_count[target_key] = 0

    def record_failure(self, target_key: str) -> None:
        """Record a failed request.

        Args:
            target_key: Target identifier.
        """
        self._total_failures[target_key] += 1
        self._last_failure_time[target_key] = time.time()
        current = self._state[target_key]

        if current == CircuitState.HALF_OPEN:
            # Failure in half-open — re-open the circuit
            self._state[target_key] = CircuitState.OPEN
            self._opened_at[target_key] = time.time()
            self._success_count[target_key] = 0

        elif current == CircuitState.CLOSED:
            self._failure_count[target_key] += 1
            if self._failure_count[target_key] >= self.config.failure_threshold:
                self._state[target_key] = CircuitState.OPEN
                self._opened_at[target_key] = time.time()

    def is_allowed(self, target_key: str) -> bool:
        """Check if a request is allowed for the target.

        Args:
            target_key: Target identifier.

        Returns:
            True if the request can proceed.
        """
        state = self._state[target_key]

        if state == CircuitState.CLOSED:
            return True

        if state == CircuitState.OPEN:
            # Check if timeout has elapsed
            opened_at = self._opened_at.get(target_key, 0)
            if time.time() - opened_at >= self.config.timeout:
                self._state[target_key] = CircuitState.HALF_OPEN
                self._success_count[target_key] = 0
                return True
            return False

        # HALF_OPEN — allow limited requests
        return True

    def reset(self, target_key: Optional[str] = None) -> None:
        """Reset circuit breaker state.

        Args:
            target_key: Specific target or all.
        """
        if target_key:
            self._state[target_key] = CircuitState.CLOSED
            self._failure_count[target_key] = 0
            self._success_count[target_key] = 0
        else:
            self._state.clear()
            self._failure_count.clear()
            self._success_count.clear()
            self._last_failure_time.clear()
            self._opened_at.clear()

    def get_state(self, target_key: str) -> CircuitState:
        """Get current circuit state for a target.

        Args:
            target_key: Target identifier.

        Returns:
            Current circuit state.
        """
        return self._state.get(target_key, CircuitState.CLOSED)

    def get_stats(self, target_key: str) -> Dict[str, Any]:
        """Get statistics for a target.

        Args:
            target_key: Target identifier.

        Returns:
            Dict with failure counts, state, etc.
        """
        return {
            "state": self.get_state(target_key).value,
            "consecutive_failures": self._failure_count.get(target_key, 0),
            "total_failures": self._total_failures.get(target_key, 0),
            "total_successes": self._total_successes.get(target_key, 0),
            "last_failure": self._last_failure_time.get(target_key),
        }


# ──────────────────────────────────────────────────────────────────
# Fallback Manager
# ──────────────────────────────────────────────────────────────────


class FallbackManager:
    """Manages fallback chains and degradation strategies.

    When a primary provider fails, the fallback manager:
    1. Checks the circuit breaker
    2. Tries retries on the same provider
    3. Falls back to the next provider in the chain
    4. Tracks all fallback actions for monitoring
    """

    def __init__(
        self,
        default_max_retries: int = 2,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        fallback_chains: Optional[Dict[str, List[FallbackChainEntry]]] = None,
    ):
        """Initialize the fallback manager.

        Args:
            default_max_retries: Default retry count per provider.
            circuit_breaker_config: Circuit breaker config.
            fallback_chains: Mapping of primary target -> fallback chain.
        """
        self.default_max_retries = default_max_retries
        self.circuit_breaker = CircuitBreaker(circuit_breaker_config)
        self.fallback_chains: Dict[str, List[FallbackChainEntry]] = fallback_chains or {}
        self._fallback_history: List[FallbackAction] = []
        self._max_history = 1000

    def set_fallback_chain(
        self,
        primary_key: str,
        chain: List[FallbackChainEntry],
    ) -> None:
        """Set the fallback chain for a primary target.

        Args:
            primary_key: Primary target key.
            chain: Ordered list of fallback entries.
        """
        self.fallback_chains[primary_key] = chain

    def add_fallback(
        self,
        primary_key: str,
        provider: str,
        model: str,
        weight: float = 1.0,
        max_retries: int = 2,
    ) -> None:
        """Add a single fallback to a chain.

        Args:
            primary_key: Primary target key.
            provider: Fallback provider.
            model: Fallback model.
            weight: Priority weight.
            max_retries: Max retries for this fallback.
        """
        if primary_key not in self.fallback_chains:
            self.fallback_chains[primary_key] = []
        self.fallback_chains[primary_key].append(
            FallbackChainEntry(
                provider=provider,
                model=model,
                weight=weight,
                max_retries=max_retries,
            )
        )

    def get_fallback_targets(self, primary_key: str) -> List[FallbackChainEntry]:
        """Get the fallback chain for a primary target, sorted by weight descending.

        Args:
            primary_key: Primary target key.

        Returns:
            Ordered list of fallback entries.
        """
        chain = self.fallback_chains.get(primary_key, [])
        return sorted(chain, key=lambda x: x.weight, reverse=True)

    def is_target_available(self, target_key: str) -> bool:
        """Check if a target is available (circuit not open).

        Args:
            target_key: Target identifier.

        Returns:
            True if the target can be used.
        """
        return self.circuit_breaker.is_allowed(target_key)

    async def execute_with_fallback(
        self,
        primary_key: str,
        execution_fn: Callable[[str], Any],
        fallback_targets: Optional[List[FallbackChainEntry]] = None,
        base_retries: int = 2,
        on_fallback: Optional[Callable[[FallbackAction], None]] = None,
    ) -> Any:
        """Execute a function with automatic fallback.

        Args:
            primary_key: Primary target key.
            execution_fn: Async function that takes a target_key and returns a result.
            fallback_targets: Override fallback chain.
            base_retries: Override default retry count.
            on_fallback: Callback invoked on each fallback action.

        Returns:
            Execution result.

        Raises:
            RuntimeError: If all targets (primary + fallbacks) fail.
        """
        targets = fallback_targets or self.get_fallback_targets(primary_key)

        # Build full attempt list: primary first, then fallbacks
        full_chain = [
            FallbackChainEntry(
                provider=primary_key.split(":")[0] if ":" in primary_key else primary_key,
                model=primary_key.split(":")[1] if ":" in primary_key else "default",
                max_retries=base_retries,
            )
        ] + targets

        last_error = None
        for idx, entry in enumerate(full_chain):
            target_key = f"{entry.provider}:{entry.model}"

            # Check circuit breaker
            if not self.circuit_breaker.is_allowed(target_key):
                reason = FallbackReason.CIRCUIT_OPEN
                action = FallbackAction(
                    reason=reason,
                    from_target=primary_key,
                    to_target=target_key,
                    attempt=idx,
                    error_message="Circuit breaker is open",
                )
                self._record_fallback(action)
                if on_fallback:
                    on_fallback(action)
                continue

            # Try with retries
            for attempt in range(entry.max_retries):
                try:
                    start = time.monotonic()
                    result = await execution_fn(target_key)
                    latency = (time.monotonic() - start) * 1000

                    self.circuit_breaker.record_success(target_key)
                    return result

                except Exception as e:
                    last_error = e
                    latency = (time.monotonic() - start) * 1000
                    self.circuit_breaker.record_failure(target_key)

                    if attempt < entry.max_retries - 1:
                        # Will retry on same target
                        await asyncio.sleep(min(2 ** attempt, 30))
                        continue

                    # Moving to fallback
                    reason = self._classify_error(e)
                    action = FallbackAction(
                        reason=reason,
                        from_target=primary_key,
                        to_target=target_key,
                        attempt=idx,
                        error_message=str(e)[:200],
                        latency_ms=latency,
                    )
                    self._record_fallback(action)
                    if on_fallback:
                        on_fallback(action)

        raise RuntimeError(
            f"All targets exhausted for {primary_key}. "
            f"Last error: {last_error}"
        )

    def _classify_error(self, exc: Exception) -> FallbackReason:
        """Classify an exception into a fallback reason.

        Args:
            exc: The exception.

        Returns:
            FallbackReason enum value.
        """
        msg = str(exc).lower()
        if "timeout" in msg or "timed out" in msg:
            return FallbackReason.TIMEOUT
        if "429" in msg or "rate limit" in msg:
            return FallbackReason.RATE_LIMIT
        if "401" in msg or "unauthorized" in msg or "auth" in msg:
            return FallbackReason.AUTH_ERROR
        if "context" in msg and ("length" in msg or "exceed" in msg):
            return FallbackReason.CONTEXT_OVERFLOW
        return FallbackReason.PROVIDER_ERROR

    def _record_fallback(self, action: FallbackAction) -> None:
        """Record a fallback action in history.

        Args:
            action: The fallback action to record.
        """
        self._fallback_history.append(action)
        if len(self._fallback_history) > self._max_history:
            self._fallback_history = self._fallback_history[-self._max_history:]

    def get_history(
        self, limit: int = 50, reason: Optional[FallbackReason] = None
    ) -> List[FallbackAction]:
        """Get fallback history.

        Args:
            limit: Max entries to return.
            reason: Filter by fallback reason.

        Returns:
            List of fallback actions.
        """
        history = self._fallback_history
        if reason:
            history = [a for a in history if a.reason == reason]
        return history[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregate fallback statistics.

        Returns:
            Dict with counts, rates, circuit states.
        """
        total = len(self._fallback_history)
        reason_counts: Dict[str, int] = {}
        for action in self._fallback_history:
            reason_counts[action.reason.value] = (
                reason_counts.get(action.reason.value, 0) + 1
            )

        return {
            "total_fallbacks": total,
            "reason_breakdown": reason_counts,
            "fallback_chains": {
                k: len(v) for k, v in self.fallback_chains.items()
            },
            "circuit_states": {
                k: v.value
                for k, v in self.circuit_breaker._state.items()
            },
        }

    def clear_history(self) -> None:
        """Clear fallback history."""
        self._fallback_history.clear()

    def reset(self) -> None:
        """Reset all state — circuit breakers and history."""
        self.circuit_breaker.reset()
        self.clear_history()
