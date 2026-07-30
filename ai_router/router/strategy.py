"""
Intelligent routing strategies for selecting the optimal LLM provider and model.

Supports multiple selection algorithms:
  - Round-robin for simple load distribution
  - Weighted selection for preference-based routing
  - Lowest-latency for performance optimization
  - Lowest-cost for budget optimization
  - Semantic routing for intent-aware provider selection
"""

from __future__ import annotations

import hashlib
import heapq
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class StrategyType(str, Enum):
    """Available routing strategy types."""

    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    LOWEST_LATENCY = "lowest_latency"
    LOWEST_COST = "lowest_cost"
    SEMANTIC = "semantic"
    ADAPTIVE = "adaptive"
    RANDOM = "random"


@dataclass
class RouteTarget:
    """A routeable target — provider + model combination."""

    provider: str
    model: str
    weight: float = 1.0
    max_tokens: int = 4096
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    capabilities: List[str] = field(default_factory=lambda: ["chat"])
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Unique key for this route target."""
        return f"{self.provider}:{self.model}"


@dataclass
class RouteRequest:
    """A routing request with optional constraints."""

    messages: List[Dict[str, Any]] = field(default_factory=list)
    max_tokens: Optional[int] = None
    required_capabilities: List[str] = field(default_factory=list)
    preferred_provider: Optional[str] = None
    preferred_model: Optional[str] = None
    max_cost: Optional[float] = None
    max_latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RouteDecision:
    """The result of a routing decision."""

    target: RouteTarget
    strategy: StrategyType
    reason: str = ""
    confidence: float = 1.0
    alternatives: List[RouteTarget] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────
# Performance Tracking
# ──────────────────────────────────────────────────────────────────


class PerformanceTracker:
    """Tracks latency, success rates, and costs for route targets.

    Uses exponential moving averages for smooth performance estimates.
    """

    def __init__(
        self,
        alpha: float = 0.2,
        window_size: int = 100,
        decay_factor: float = 0.95,
    ):
        """Initialize the performance tracker.

        Args:
            alpha: EMA smoothing factor for latency (0-1).
            window_size: Max number of recent samples to keep.
            decay_factor: Weight decay for older successes.
        """
        self.alpha = alpha
        self.window_size = window_size
        self.decay_factor = decay_factor

        # Per-target metrics
        self._latency_ema: Dict[str, float] = {}
        self._success_counts: Dict[str, int] = defaultdict(int)
        self._failure_counts: Dict[str, int] = defaultdict(int)
        self._total_costs: Dict[str, float] = defaultdict(float)
        self._latency_history: Dict[str, List[float]] = defaultdict(list)
        self._last_used: Dict[str, float] = {}

    def record_success(
        self,
        target_key: str,
        latency_ms: float,
        cost: float = 0.0,
    ) -> None:
        """Record a successful request.

        Args:
            target_key: Route target key.
            latency_ms: Request latency in milliseconds.
            cost: Request cost in dollars.
        """
        self._success_counts[target_key] += 1

        # Update EMA latency
        if target_key in self._latency_ema:
            self._latency_ema[target_key] = (
                self.alpha * latency_ms
                + (1 - self.alpha) * self._latency_ema[target_key]
            )
        else:
            self._latency_ema[target_key] = latency_ms

        # Track history
        history = self._latency_history[target_key]
        history.append(latency_ms)
        if len(history) > self.window_size:
            history.pop(0)

        self._total_costs[target_key] += cost
        self._last_used[target_key] = time.time()

    def record_failure(self, target_key: str) -> None:
        """Record a failed request.

        Args:
            target_key: Route target key.
        """
        self._failure_counts[target_key] += 1
        self._last_used[target_key] = time.time()

    def get_latency(self, target_key: str) -> Optional[float]:
        """Get the EMA latency for a target."""
        return self._latency_ema.get(target_key)

    def get_success_rate(self, target_key: str) -> float:
        """Get the success rate (0-1) for a target."""
        total = self._success_counts[target_key] + self._failure_counts[target_key]
        if total == 0:
            return 1.0
        return self._success_counts[target_key] / total

    def get_p95_latency(self, target_key: str) -> Optional[float]:
        """Get the 95th percentile latency."""
        history = self._latency_history.get(target_key, [])
        if not history:
            return None
        return float(np.percentile(history, 95))

    def get_p50_latency(self, target_key: str) -> Optional[float]:
        """Get the median latency."""
        history = self._latency_history.get(target_key, [])
        if not history:
            return None
        return float(np.percentile(history, 50))

    def get_total_cost(self, target_key: str) -> float:
        """Get total cost for a target."""
        return self._total_costs.get(target_key, 0.0)

    def get_health_score(self, target_key: str) -> float:
        """Calculate a composite health score (0-100).

        Factors: success rate, latency relative to best, recency.
        """
        success_rate = self.get_success_rate(target_key)
        if success_rate == 0:
            return 0.0

        latency = self.get_latency(target_key)
        latency_score = 100.0
        if latency and self._latency_ema:
            best_latency = min(self._latency_ema.values())
            if best_latency > 0:
                latency_score = min(100.0, (best_latency / max(latency, 1)) * 100)

        return success_rate * 70 + latency_score * 0.3

    def reset(self, target_key: Optional[str] = None) -> None:
        """Reset metrics for a target or all targets.

        Args:
            target_key: Specific target to reset, or None for all.
        """
        if target_key:
            self._latency_ema.pop(target_key, None)
            self._success_counts.pop(target_key, None)
            self._failure_counts.pop(target_key, None)
            self._total_costs.pop(target_key, None)
            self._latency_history.pop(target_key, None)
            self._last_used.pop(target_key, None)
        else:
            self._latency_ema.clear()
            self._success_counts.clear()
            self._failure_counts.clear()
            self._total_costs.clear()
            self._latency_history.clear()
            self._last_used.clear()


# ──────────────────────────────────────────────────────────────────
# Strategy Base
# ──────────────────────────────────────────────────────────────────


class Strategy(ABC):
    """Abstract base class for routing strategies.

    Each strategy implements the `select` method to choose a route target
    from a list of available targets based on a routing request and
    performance history.
    """

    def __init__(self, tracker: Optional[PerformanceTracker] = None):
        """Initialize the strategy.

        Args:
            tracker: Optional performance tracker for data-driven decisions.
        """
        self.tracker = tracker or PerformanceTracker()

    @abstractmethod
    def select(
        self,
        targets: List[RouteTarget],
        request: RouteRequest,
    ) -> RouteDecision:
        """Select the best route target.

        Args:
            targets: Available route targets.
            request: Routing request with constraints.

        Returns:
            A RouteDecision with the selected target.
        """
        ...

    def filter_targets(
        self,
        targets: List[RouteTarget],
        request: RouteRequest,
    ) -> List[RouteTarget]:
        """Filter targets based on request constraints.

        Args:
            targets: Available targets.
            request: Request with constraints.

        Returns:
            Filtered list of eligible targets.
        """
        eligible = list(targets)

        # Filter by capabilities
        if request.required_capabilities:
            eligible = [
                t for t in eligible
                if all(c in t.capabilities for c in request.required_capabilities)
            ]

        # Filter by max tokens
        if request.max_tokens is not None:
            eligible = [t for t in eligible if t.max_tokens >= request.max_tokens]

        # Filter by preferred provider
        if request.preferred_provider:
            preferred = [t for t in eligible if t.provider == request.preferred_provider]
            if preferred:
                eligible = preferred

        # Filter by preferred model
        if request.preferred_model:
            preferred = [t for t in eligible if t.model == request.preferred_model]
            if preferred:
                eligible = preferred

        return eligible

    def record_result(
        self,
        target_key: str,
        success: bool,
        latency_ms: float = 0.0,
        cost: float = 0.0,
    ) -> None:
        """Record the result of a routing decision.

        Args:
            target_key: The target key.
            success: Whether the request succeeded.
            latency_ms: Request latency.
            cost: Request cost.
        """
        if success:
            self.tracker.record_success(target_key, latency_ms, cost)
        else:
            self.tracker.record_failure(target_key)


# ──────────────────────────────────────────────────────────────────
# Round-Robin Strategy
# ──────────────────────────────────────────────────────────────────


class RoundRobinStrategy(Strategy):
    """Simple round-robin strategy that cycles through targets.

    Distributes requests evenly across all available targets, ignoring
    performance metrics. Best suited for load balancing when all targets
    have similar characteristics.
    """

    def __init__(self, tracker: Optional[PerformanceTracker] = None, seed: int = 0):
        """Initialize the round-robin strategy.

        Args:
            tracker: Optional performance tracker.
            seed: Starting index for the round-robin cycle.
        """
        super().__init__(tracker)
        self._counters: Dict[str, int] = defaultdict(lambda: seed)

    def select(
        self,
        targets: List[RouteTarget],
        request: RouteRequest,
    ) -> RouteDecision:
        """Select the next target in round-robin order.

        Args:
            targets: Available targets.
            request: Routing request.

        Returns:
            RouteDecision with the selected target.

        Raises:
            ValueError: If no eligible targets are available.
        """
        eligible = self.filter_targets(targets, request)

        if not eligible:
            raise ValueError("No eligible targets available")

        if len(eligible) == 1:
            return RouteDecision(
                target=eligible[0],
                strategy=StrategyType.ROUND_ROBIN,
                reason="Only one eligible target",
            )

        # Group counter key by capability set for fairness
        counter_key = ",".join(request.required_capabilities) or "default"
        idx = self._counters[counter_key] % len(eligible)
        self._counters[counter_key] += 1

        selected = eligible[idx]

        # List alternatives
        alternatives = [t for t in eligible if t.key != selected.key]

        return RouteDecision(
            target=selected,
            strategy=StrategyType.ROUND_ROBIN,
            reason=f"Round-robin selection (turn {self._counters[counter_key]})",
            alternatives=alternatives,
        )


# ──────────────────────────────────────────────────────────────────
# Weighted Strategy
# ──────────────────────────────────────────────────────────────────


class WeightedStrategy(Strategy):
    """Weighted random selection strategy.

    Selects targets with probability proportional to their configured weights.
    Higher weight = more likely to be selected.
    """

    def select(
        self,
        targets: List[RouteTarget],
        request: RouteRequest,
    ) -> RouteDecision:
        """Select a target based on weighted random sampling.

        Args:
            targets: Available targets with weights.
            request: Routing request.

        Returns:
            RouteDecision with the selected target.

        Raises:
            ValueError: If no eligible targets or all weights are zero.
        """
        eligible = self.filter_targets(targets, request)

        if not eligible:
            raise ValueError("No eligible targets available")

        if len(eligible) == 1:
            return RouteDecision(
                target=eligible[0],
                strategy=StrategyType.WEIGHTED,
                reason="Only one eligible target",
            )

        weights = [t.weight for t in eligible]
        total_weight = sum(weights)

        if total_weight <= 0:
            raise ValueError("All target weights are zero or negative")

        # Weighted random selection
        rand = random.random() * total_weight
        cumulative = 0.0
        selected_idx = 0
        for i, w in enumerate(weights):
            cumulative += w
            if rand <= cumulative:
                selected_idx = i
                break

        selected = eligible[selected_idx]
        alternatives = [t for t in eligible if t.key != selected.key]

        return RouteDecision(
            target=selected,
            strategy=StrategyType.WEIGHTED,
            reason=f"Weighted selection (weight={selected.weight:.2f})",
            alternatives=alternatives,
        )


# ──────────────────────────────────────────────────────────────────
# Lowest Latency Strategy
# ──────────────────────────────────────────────────────────────────


class LowestLatencyStrategy(Strategy):
    """Selects the target with the lowest estimated latency.

    Uses EMA latency from the performance tracker to choose the fastest
    available target. Includes epsilon-greedy exploration for discovering
    better performers.
    """

    def __init__(
        self,
        tracker: Optional[PerformanceTracker] = None,
        epsilon: float = 0.1,
        min_samples: int = 3,
    ):
        """Initialize the lowest-latency strategy.

        Args:
            tracker: Performance tracker for latency data.
            epsilon: Exploration rate (probability of random selection).
            min_samples: Minimum samples before trusting latency estimate.
        """
        super().__init__(tracker)
        self.epsilon = epsilon
        self.min_samples = min_samples

    def select(
        self,
        targets: List[RouteTarget],
        request: RouteRequest,
    ) -> RouteDecision:
        """Select the target with the lowest expected latency.

        Args:
            targets: Available targets.
            request: Routing request.

        Returns:
            RouteDecision with the selected target.
        """
        eligible = self.filter_targets(targets, request)

        if not eligible:
            raise ValueError("No eligible targets available")

        if len(eligible) == 1:
            return RouteDecision(
                target=eligible[0],
                strategy=StrategyType.LOWEST_LATENCY,
                reason="Only one eligible target",
            )

        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            selected = random.choice(eligible)
            return RouteDecision(
                target=selected,
                strategy=StrategyType.LOWEST_LATENCY,
                reason="Exploration (epsilon-greedy)",
                alternatives=[t for t in eligible if t.key != selected.key],
            )

        # Select lowest latency
        latencies = []
        for t in eligible:
            lat = self.tracker.get_latency(t.key)
            count = (
                self.tracker._success_counts[t.key]
                + self.tracker._failure_counts[t.key]
            )
            if lat is not None and count >= self.min_samples:
                latencies.append((lat, t))
            else:
                # Unknown targets get medium priority (for exploration)
                latencies.append((float("inf"), t))

        latencies.sort(key=lambda x: x[0])

        # If all are unknown, pick randomly
        if latencies[0][0] == float("inf"):
            selected = random.choice(eligible)
            reason = "All targets untested — random selection"
        else:
            selected = latencies[0][1]
            reason = f"Lowest EMA latency: {latencies[0][0]:.0f}ms"

        alternatives = [t for t in eligible if t.key != selected.key]

        return RouteDecision(
            target=selected,
            strategy=StrategyType.LOWEST_LATENCY,
            reason=reason,
            alternatives=alternatives,
        )


# ──────────────────────────────────────────────────────────────────
# Lowest Cost Strategy
# ──────────────────────────────────────────────────────────────────


class LowestCostStrategy(Strategy):
    """Selects the target with the lowest estimated cost.

    Estimates cost based on configured cost-per-token and expected token usage.
    """

    def __init__(
        self,
        tracker: Optional[PerformanceTracker] = None,
        estimated_input_tokens: int = 1000,
        estimated_output_tokens: int = 500,
    ):
        """Initialize the lowest-cost strategy.

        Args:
            tracker: Performance tracker.
            estimated_input_tokens: Expected prompt tokens for cost estimation.
            estimated_output_tokens: Expected completion tokens for cost estimation.
        """
        super().__init__(tracker)
        self.estimated_input_tokens = estimated_input_tokens
        self.estimated_output_tokens = estimated_output_tokens

    def select(
        self,
        targets: List[RouteTarget],
        request: RouteRequest,
    ) -> RouteDecision:
        """Select the target with the lowest estimated cost.

        Args:
            targets: Available targets.
            request: Routing request.

        Returns:
            RouteDecision with the selected target.
        """
        eligible = self.filter_targets(targets, request)

        if not eligible:
            raise ValueError("No eligible targets available")

        if len(eligible) == 1:
            return RouteDecision(
                target=eligible[0],
                strategy=StrategyType.LOWEST_COST,
                reason="Only one eligible target",
            )

        # Calculate estimated costs
        costs = []
        for t in eligible:
            estimated_cost = (
                self.estimated_input_tokens / 1000 * t.cost_per_1k_input
                + self.estimated_output_tokens / 1000 * t.cost_per_1k_output
            )
            # Blend with actual historical cost if available
            historical = self.tracker.get_total_cost(t.key)
            if historical > 0:
                count = self.tracker._success_counts.get(t.key, 1)
                avg_historical = historical / max(count, 1)
                estimated_cost = 0.3 * estimated_cost + 0.7 * avg_historical
            costs.append((estimated_cost, t))

        costs.sort(key=lambda x: x[0])
        selected = costs[0][1]

        alternatives = [t for t in eligible if t.key != selected.key]

        return RouteDecision(
            target=selected,
            strategy=StrategyType.LOWEST_COST,
            reason=f"Lowest estimated cost: ${costs[0][0]:.6f}",
            alternatives=alternatives,
        )


# ──────────────────────────────────────────────────────────────────
# Semantic Router
# ──────────────────────────────────────────────────────────────────


class SemanticRouter(Strategy):
    """Intent-aware semantic routing.

    Routes requests to the most appropriate provider/model based on
    the semantic similarity between the request content and target
    descriptions. Each target can have an embedding description that
    represents the types of tasks it excels at.
    """

    def __init__(
        self,
        tracker: Optional[PerformanceTracker] = None,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        target_embeddings: Optional[Dict[str, List[float]]] = None,
        target_descriptions: Optional[Dict[str, str]] = None,
        similarity_threshold: float = 0.5,
    ):
        """Initialize the semantic router.

        Args:
            tracker: Performance tracker.
            embed_fn: Function to generate embeddings for text.
            target_embeddings: Pre-computed embeddings for targets.
            target_descriptions: Text descriptions of target capabilities.
            similarity_threshold: Minimum similarity to consider a match.
        """
        super().__init__(tracker)
        self.embed_fn = embed_fn or self._default_embed_fn
        self.target_embeddings: Dict[str, List[float]] = target_embeddings or {}
        self.target_descriptions: Dict[str, str] = target_descriptions or {}
        self.similarity_threshold = similarity_threshold
        self._embedding_cache: Dict[str, List[float]] = {}

    @staticmethod
    def _default_embed_fn(text: str) -> List[float]:
        """Default embedding function using simple hash-based approach.

        In production, replace with an actual embedding model.
        """
        # Simple n-gram hash embedding for demonstration
        n = 3
        dim = 128
        vec = [0.0] * dim
        text_lower = text.lower()
        for i in range(len(text_lower) - n + 1):
            ngram = text_lower[i : i + n]
            h = int(hashlib.md5(ngram.encode()).hexdigest(), 16) % dim
            vec[h] += 1.0
        # Normalize
        norm = np.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def set_target_description(self, target_key: str, description: str) -> None:
        """Set the capability description for a target.

        Args:
            target_key: Route target key.
            description: Natural language description of target strengths.
        """
        self.target_descriptions[target_key] = description
        # Clear cached embedding
        self._embedding_cache.pop(target_key, None)
        self.target_embeddings.pop(target_key, None)

    def get_target_embedding(self, target_key: str) -> Optional[List[float]]:
        """Get or compute the embedding for a target.

        Args:
            target_key: Route target key.

        Returns:
            Embedding vector or None if no description is set.
        """
        if target_key in self.target_embeddings:
            return self.target_embeddings[target_key]

        if target_key in self._embedding_cache:
            return self._embedding_cache[target_key]

        description = self.target_descriptions.get(target_key)
        if description is None:
            return None

        embedding = self.embed_fn(description)
        self._embedding_cache[target_key] = embedding
        return embedding

    def get_query_embedding(self, request: RouteRequest) -> List[float]:
        """Generate embedding for a routing request.

        Concatenates all message content into a single query string.

        Args:
            request: The routing request.

        Returns:
            Embedding vector for the request.
        """
        texts = []
        for msg in request.messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        texts.append(part.get("text", str(part)))
                    else:
                        texts.append(str(part))

        # Use metadata tags too
        tags = request.metadata.get("tags", [])
        if tags:
            texts.extend(tags)

        query = " ".join(texts)
        query_hash = hashlib.md5(query.encode()).hexdigest()

        if query_hash in self._embedding_cache:
            return self._embedding_cache[query_hash]

        embedding = self.embed_fn(query)
        self._embedding_cache[query_hash] = embedding
        return embedding

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Cosine similarity (0-1 range typically).
        """
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = np.sqrt(sum(x * x for x in a))
        norm_b = np.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def select(
        self,
        targets: List[RouteTarget],
        request: RouteRequest,
    ) -> RouteDecision:
        """Select the semantically best matching target.

        Args:
            targets: Available targets.
            request: Routing request with message content.

        Returns:
            RouteDecision with the best matching target.
        """
        # First filter by hard constraints
        eligible = self.filter_targets(targets, request)

        if not eligible:
            raise ValueError("No eligible targets after filtering")

        if len(eligible) == 1:
            return RouteDecision(
                target=eligible[0],
                strategy=StrategyType.SEMANTIC,
                reason="Only one eligible target",
            )

        query_embedding = self.get_query_embedding(request)

        # Compute similarities
        scored: List[Tuple[float, RouteTarget]] = []
        no_embedding_targets: List[RouteTarget] = []

        for t in eligible:
            target_emb = self.get_target_embedding(t.key)
            if target_emb is not None:
                sim = self._cosine_similarity(query_embedding, target_emb)
                scored.append((sim, t))
            else:
                no_embedding_targets.append(t)

        if not scored:
            # No embeddings available — fall back to weighted
            weighted = WeightedStrategy(tracker=self.tracker)
            return weighted.select(eligible, request)

        scored.sort(key=lambda x: x[0], reverse=True)
        best_sim, selected = scored[0]

        # Check threshold
        if best_sim < self.similarity_threshold and no_embedding_targets:
            # Fall back to any target without embeddings
            import random
            selected = random.choice(no_embedding_targets)
            reason = (
                f"Low similarity ({best_sim:.3f}) — "
                f"falling back to untagged target"
            )
        else:
            reason = f"Best semantic match (similarity={best_sim:.3f})"

        alternatives = [t for _, t in scored[1:]] + no_embedding_targets

        return RouteDecision(
            target=selected,
            strategy=StrategyType.SEMANTIC,
            reason=reason,
            confidence=max(0.0, best_sim),
            alternatives=alternatives[:5],
        )


# ──────────────────────────────────────────────────────────────────
# Adaptive Strategy
# ──────────────────────────────────────────────────────────────────


class AdaptiveStrategy(Strategy):
    """Multi-factor adaptive strategy that balances latency, cost, and success rate.

    Uses a weighted scoring system combining:
    - Success rate (reliability)
    - Latency (performance)
    - Cost (economy)

    Weights are configurable to prioritize different factors.
    """

    def __init__(
        self,
        tracker: Optional[PerformanceTracker] = None,
        reliability_weight: float = 0.5,
        latency_weight: float = 0.3,
        cost_weight: float = 0.2,
        min_samples: int = 5,
    ):
        """Initialize the adaptive strategy.

        Args:
            tracker: Performance tracker.
            reliability_weight: Weight for success rate (0-1).
            latency_weight: Weight for latency score (0-1).
            cost_weight: Weight for cost score (0-1).
            min_samples: Minimum samples before trusting metrics.
        """
        super().__init__(tracker)
        total = reliability_weight + latency_weight + cost_weight
        self.reliability_weight = reliability_weight / total
        self.latency_weight = latency_weight / total
        self.cost_weight = cost_weight / total
        self.min_samples = min_samples

    def select(
        self,
        targets: List[RouteTarget],
        request: RouteRequest,
    ) -> RouteDecision:
        """Select the best target using multi-factor scoring.

        Args:
            targets: Available targets.
            request: Routing request.

        Returns:
            RouteDecision with the best overall target.
        """
        eligible = self.filter_targets(targets, request)

        if not eligible:
            raise ValueError("No eligible targets available")

        if len(eligible) == 1:
            return RouteDecision(
                target=eligible[0],
                strategy=StrategyType.ADAPTIVE,
                reason="Only one eligible target",
            )

        # Collect metrics
        all_latencies = []
        all_costs = []
        for t in eligible:
            lat = self.tracker.get_latency(t.key)
            if lat is not None:
                all_latencies.append(lat)
            historical = self.tracker.get_total_cost(t.key)
            count = max(self.tracker._success_counts.get(t.key, 1), 1)
            avg_cost = historical / count if historical > 0 else (
                t.cost_per_1k_input * 1 + t.cost_per_1k_output * 0.5
            )
            all_costs.append(avg_cost)

        max_latency = max(all_latencies) if all_latencies else 1000.0
        max_cost = max(all_costs) if all_costs else 0.001

        scored = []
        for t in eligible:
            # Reliability score
            success_rate = self.tracker.get_success_rate(t.key)
            count = (
                self.tracker._success_counts[t.key]
                + self.tracker._failure_counts[t.key]
            )
            if count < self.min_samples:
                # Unknown — assume decent but not perfect
                reliability = 0.5
            else:
                reliability = success_rate

            # Latency score (normalized, higher is better)
            lat = self.tracker.get_latency(t.key)
            if lat is None:
                lat_score = 0.5
            elif max_latency == 0:
                lat_score = 1.0
            else:
                lat_score = 1.0 - (lat / max_latency)

            # Cost score (normalized, higher is better = cheaper)
            historical = self.tracker.get_total_cost(t.key)
            count2 = max(self.tracker._success_counts.get(t.key, 1), 1)
            avg_cost = historical / count2 if historical > 0 else (
                t.cost_per_1k_input * 1 + t.cost_per_1k_output * 0.5
            )
            if max_cost == 0:
                cost_score = 1.0
            else:
                cost_score = 1.0 - (avg_cost / max_cost)

            composite = (
                self.reliability_weight * reliability
                + self.latency_weight * lat_score
                + self.cost_weight * cost_score
            )
            scored.append((composite, t))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = scored[0][1]

        reason = (
            f"Adaptive selection: reliability={self.reliability_weight:.0%}, "
            f"latency={self.latency_weight:.0%}, cost={self.cost_weight:.0%}"
        )

        alternatives = [t for _, t in scored[1:]]

        return RouteDecision(
            target=selected,
            strategy=StrategyType.ADAPTIVE,
            reason=reason,
            confidence=scored[0][0],
            alternatives=alternatives,
        )


# ──────────────────────────────────────────────────────────────────
# Random Strategy
# ──────────────────────────────────────────────────────────────────


class RandomStrategy(Strategy):
    """Simple random selection — useful as a baseline."""

    def select(
        self,
        targets: List[RouteTarget],
        request: RouteRequest,
    ) -> RouteDecision:
        """Select a random eligible target.

        Args:
            targets: Available targets.
            request: Routing request.

        Returns:
            RouteDecision with a randomly selected target.
        """
        eligible = self.filter_targets(targets, request)

        if not eligible:
            raise ValueError("No eligible targets available")

        selected = random.choice(eligible)
        alternatives = [t for t in eligible if t.key != selected.key]

        return RouteDecision(
            target=selected,
            strategy=StrategyType.RANDOM,
            reason="Random selection",
            alternatives=alternatives,
        )


# ──────────────────────────────────────────────────────────────────
# Strategy Factory
# ──────────────────────────────────────────────────────────────────


_STRATEGY_REGISTRY: Dict[StrategyType, type] = {
    StrategyType.ROUND_ROBIN: RoundRobinStrategy,
    StrategyType.WEIGHTED: WeightedStrategy,
    StrategyType.LOWEST_LATENCY: LowestLatencyStrategy,
    StrategyType.LOWEST_COST: LowestCostStrategy,
    StrategyType.SEMANTIC: SemanticRouter,
    StrategyType.ADAPTIVE: AdaptiveStrategy,
    StrategyType.RANDOM: RandomStrategy,
}


def get_strategy_class(strategy_type: StrategyType) -> type:
    """Get a strategy class by type.

    Args:
        strategy_type: The strategy type enum.

    Returns:
        Strategy class.

    Raises:
        ValueError: If strategy type is unknown.
    """
    if strategy_type not in _STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy type: {strategy_type}. "
            f"Available: {list(_STRATEGY_REGISTRY.keys())}"
        )
    return _STRATEGY_REGISTRY[strategy_type]


def create_strategy(
    strategy_type: StrategyType,
    tracker: Optional[PerformanceTracker] = None,
    **kwargs: Any,
) -> Strategy:
    """Factory to create a strategy instance.

    Args:
        strategy_type: Type of strategy to create.
        tracker: Shared performance tracker.
        **kwargs: Strategy-specific arguments.

    Returns:
        Strategy instance.

    Raises:
        ValueError: If strategy type is unknown.
    """
    cls = get_strategy_class(strategy_type)
    return cls(tracker=tracker, **kwargs)


def register_strategy(strategy_type: StrategyType, strategy_cls: type) -> None:
    """Register a custom strategy.

    Args:
        strategy_type: Enum value for the strategy.
        strategy_cls: Strategy subclass.
    """
    _STRATEGY_REGISTRY[strategy_type] = strategy_cls
