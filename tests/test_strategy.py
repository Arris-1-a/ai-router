"""Tests for routing strategy module."""

import pytest
from ai_router.router.strategy import (
    AdaptiveStrategy,
    LowestCostStrategy,
    LowestLatencyStrategy,
    PerformanceTracker,
    RandomStrategy,
    RouteDecision,
    RouteRequest,
    RouteTarget,
    RoundRobinStrategy,
    SemanticRouter,
    StrategyType,
    WeightedStrategy,
    create_strategy,
)


@pytest.fixture
def targets():
    """Create a set of test route targets."""
    return [
        RouteTarget(provider="openai", model="gpt-4o", weight=2.0,
                    cost_per_1k_input=0.005, cost_per_1k_output=0.015, max_tokens=4096),
        RouteTarget(provider="openai", model="gpt-4o-mini", weight=3.0,
                    cost_per_1k_input=0.00015, cost_per_1k_output=0.0006, max_tokens=4096),
        RouteTarget(provider="anthropic", model="claude-3-haiku", weight=1.5,
                    cost_per_1k_input=0.00025, cost_per_1k_output=0.00125, max_tokens=4096),
        RouteTarget(provider="deepseek", model="deepseek-chat", weight=1.0,
                    cost_per_1k_input=0.00014, cost_per_1k_output=0.00028, max_tokens=4096),
    ]


@pytest.fixture
def request_data():
    """Create a test route request."""
    return RouteRequest(
        messages=[{"role": "user", "content": "Hello, world!"}],
    )


class TestPerformanceTracker:
    """Tests for PerformanceTracker."""

    def test_record_success(self):
        tracker = PerformanceTracker()
        tracker.record_success("openai:gpt-4o", latency_ms=500, cost=0.01)
        tracker.record_success("openai:gpt-4o", latency_ms=300, cost=0.005)

        latency = tracker.get_latency("openai:gpt-4o")
        assert latency is not None
        # EMA should be between two values
        assert 300 < latency < 500

    def test_record_failure(self):
        tracker = PerformanceTracker()
        tracker.record_success("test", 100)
        tracker.record_failure("test")
        rate = tracker.get_success_rate("test")
        assert rate == 0.5

    def test_success_rate_new_target(self):
        tracker = PerformanceTracker()
        rate = tracker.get_success_rate("new_target")
        assert rate == 1.0

    def test_reset(self):
        tracker = PerformanceTracker()
        tracker.record_success("test", 100)
        tracker.reset("test")
        assert tracker.get_latency("test") is None


class TestRoundRobinStrategy:
    """Tests for RoundRobinStrategy."""

    def test_cycles_through_targets(self, targets, request_data):
        strategy = RoundRobinStrategy()
        decisions = [strategy.select(targets, request_data) for _ in range(4)]

        # Should visit each target
        visited = {d.target.key for d in decisions}
        assert len(visited) == 4

    def test_single_target(self, request_data):
        single_target = [RouteTarget(provider="openai", model="gpt-4o")]
        strategy = RoundRobinStrategy()
        decision = strategy.select(single_target, request_data)
        assert decision.target.key == "openai:gpt-4o"
        assert decision.strategy == StrategyType.ROUND_ROBIN

    def test_no_targets(self, request_data):
        strategy = RoundRobinStrategy()
        with pytest.raises(ValueError, match="No eligible targets"):
            strategy.select([], request_data)

    def test_filters_by_capabilities(self, request_data):
        targets = [
            RouteTarget(provider="openai", model="gpt-4o", capabilities=["chat", "vision"]),
            RouteTarget(provider="openai", model="gpt-4o-mini", capabilities=["chat"]),
        ]
        request_data.required_capabilities = ["vision"]
        strategy = RoundRobinStrategy()
        decision = strategy.select(targets, request_data)
        assert decision.target.model == "gpt-4o"

    def test_filter_preferred_provider(self, request_data):
        targets = [
            RouteTarget(provider="openai", model="gpt-4o"),
            RouteTarget(provider="anthropic", model="claude-3-haiku"),
        ]
        request_data.preferred_provider = "anthropic"
        strategy = RoundRobinStrategy()
        decision = strategy.select(targets, request_data)
        assert decision.target.provider == "anthropic"


class TestWeightedStrategy:
    """Tests for WeightedStrategy."""

    def test_higher_weight_more_likely(self, targets, request_data):
        strategy = WeightedStrategy()
        # Run many times and check distribution
        counts = {}
        for _ in range(1000):
            decision = strategy.select(targets, request_data)
            key = decision.target.key
            counts[key] = counts.get(key, 0) + 1

        # openai:gpt-4o-mini has highest weight (3.0), should be selected most
        max_key = max(counts, key=counts.get)
        assert "gpt-4o-mini" in max_key

    def test_zero_weights(self, request_data):
        targets = [
            RouteTarget(provider="a", model="m", weight=0),
            RouteTarget(provider="b", model="m", weight=0),
        ]
        strategy = WeightedStrategy()
        with pytest.raises(ValueError):
            strategy.select(targets, request_data)


class TestLowestLatencyStrategy:
    """Tests for LowestLatencyStrategy."""

    def test_selects_lowest_latency(self):
        tracker = PerformanceTracker()
        targets = [
            RouteTarget(provider="a", model="fast"),
            RouteTarget(provider="b", model="slow"),
        ]
        tracker.record_success("a:fast", latency_ms=100)
        tracker.record_success("a:fast", latency_ms=120)
        tracker.record_success("a:fast", latency_ms=110)
        tracker.record_success("b:slow", latency_ms=500)
        tracker.record_success("b:slow", latency_ms=520)
        tracker.record_success("b:slow", latency_ms=480)

        strategy = LowestLatencyStrategy(tracker=tracker, epsilon=0.0, min_samples=3)
        decision = strategy.select(targets, RouteRequest())
        assert "fast" in decision.target.model


class TestLowestCostStrategy:
    """Tests for LowestCostStrategy."""

    def test_selects_cheapest(self, targets, request_data):
        strategy = LowestCostStrategy(estimated_input_tokens=1000, estimated_output_tokens=500)
        decision = strategy.select(targets, request_data)
        # deepseek-chat has lowest cost
        assert "deepseek" in decision.target.provider


class TestSemanticRouter:
    """Tests for SemanticRouter."""

    def test_set_target_description(self):
        router = SemanticRouter()
        router.set_target_description(
            "openai:gpt-4o",
            "Best for complex reasoning and code generation tasks"
        )
        emb = router.get_target_embedding("openai:gpt-4o")
        assert emb is not None
        assert len(emb) == 128

    def test_select_with_embeddings(self):
        router = SemanticRouter()
        router.set_target_description("a:code", "code generation programming software")
        router.set_target_description("b:creative", "creative writing poetry stories")

        targets = [
            RouteTarget(provider="a", model="code"),
            RouteTarget(provider="b", model="creative"),
        ]
        request = RouteRequest(
            messages=[{"role": "user", "content": "Write a Python function to sort a list"}],
        )
        decision = router.select(targets, request)
        assert decision.target.provider == "a"  # Should route to code target

    def test_select_without_embeddings_falls_back(self):
        router = SemanticRouter()
        targets = [
            RouteTarget(provider="a", model="m1"),
            RouteTarget(provider="b", model="m2"),
        ]
        decision = router.select(targets, RouteRequest())
        assert decision.target is not None


class TestAdaptiveStrategy:
    """Tests for AdaptiveStrategy."""

    def test_uses_tracker_data(self):
        tracker = PerformanceTracker()
        targets = [
            RouteTarget(provider="a", model="good", cost_per_1k_input=0.01, cost_per_1k_output=0.03),
            RouteTarget(provider="b", model="bad", cost_per_1k_input=0.01, cost_per_1k_output=0.03),
        ]
        # Make "good" look better
        for _ in range(10):
            tracker.record_success("a:good", latency_ms=200, cost=0.001)
        for _ in range(5):
            tracker.record_success("b:bad", latency_ms=800, cost=0.005)
            tracker.record_failure("b:bad")

        strategy = AdaptiveStrategy(tracker=tracker, min_samples=5)
        decision = strategy.select(targets, RouteRequest())
        assert "good" in decision.target.model


class TestRandomStrategy:
    """Tests for RandomStrategy."""

    def test_selects_from_targets(self, targets, request_data):
        strategy = RandomStrategy()
        decision = strategy.select(targets, request_data)
        assert decision.target is not None
        assert decision.strategy == StrategyType.RANDOM


class TestStrategyFactory:
    """Tests for strategy factory."""

    def test_create_all_strategies(self):
        for stype in StrategyType:
            strategy = create_strategy(stype)
            assert strategy is not None

    def test_create_with_tracker(self):
        tracker = PerformanceTracker()
        strategy = create_strategy(StrategyType.LOWEST_LATENCY, tracker=tracker)
        assert strategy.tracker is tracker

    def test_unknown_strategy(self):
        with pytest.raises(ValueError):
            create_strategy.__wrapped__ if hasattr(create_strategy, '__wrapped__') else None
            # Test through registry
            from ai_router.router.strategy import get_strategy_class
            with pytest.raises(ValueError):
                get_strategy_class("invalid")


class TestRouteTarget:
    """Tests for RouteTarget."""

    def test_key_property(self):
        target = RouteTarget(provider="openai", model="gpt-4o")
        assert target.key == "openai:gpt-4o"

    def test_defaults(self):
        target = RouteTarget(provider="p", model="m")
        assert target.weight == 1.0
        assert target.max_tokens == 4096
        assert "chat" in target.capabilities
