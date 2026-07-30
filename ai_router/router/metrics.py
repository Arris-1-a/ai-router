"""
Metrics tracking for LLM requests — latency, cost, token usage, and success rates.

Provides real-time and historical metrics with:
  - Sliding window statistics
  - Per-provider, per-model aggregation
  - Export to various formats
  - Built-in alerting thresholds
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class MetricType(str, Enum):
    """Types of tracked metrics."""

    LATENCY = "latency"
    TOKEN_COUNT = "token_count"
    COST = "cost"
    SUCCESS_RATE = "success_rate"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"


@dataclass
class RequestMetrics:
    """Metrics for a single request."""

    request_id: str
    provider: str
    model: str
    start_time: float
    end_time: Optional[float] = None
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    success: bool = True
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    cached: bool = False
    retry_count: int = 0
    strategy: str = "unknown"
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class AggregatedMetrics:
    """Aggregated metrics over a time window."""

    provider: str
    model: str
    window_start: float
    window_end: float
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    cached_count: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.request_count == 0:
            return 1.0
        return self.success_count / self.request_count

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.request_count == 0:
            return 0.0
        return self.total_latency_ms / self.request_count

    @property
    def avg_cost(self) -> float:
        """Calculate average cost per request."""
        if self.request_count == 0:
            return 0.0
        return self.total_cost / self.request_count

    @property
    def throughput(self) -> float:
        """Calculate requests per second."""
        duration = self.window_end - self.window_start
        if duration <= 0:
            return 0.0
        return self.request_count / duration


@dataclass
class AlertRule:
    """Rule for triggering metric alerts."""

    name: str
    metric: MetricType
    condition: str  # "gt", "lt", "gte", "lte"
    threshold: float
    window_seconds: float = 300.0
    cooldown_seconds: float = 600.0
    providers: List[str] = field(default_factory=list)
    models: List[str] = field(default_factory=list)


@dataclass
class Alert:
    """A triggered alert."""

    rule_name: str
    metric: MetricType
    value: float
    threshold: float
    condition: str
    provider: str
    model: str
    timestamp: float = field(default_factory=time.time)


# ──────────────────────────────────────────────────────────────────
# Metrics Tracker
# ──────────────────────────────────────────────────────────────────


class MetricsTracker:
    """Central metrics collection and analysis system.

    Tracks per-request metrics and provides sliding window aggregations,
    alerting, and export capabilities.
    """

    def __init__(
        self,
        window_size: int = 1000,
        aggregation_window: float = 300.0,
        enable_alerts: bool = True,
    ):
        """Initialize the metrics tracker.

        Args:
            window_size: Max recent requests to keep in memory.
            aggregation_window: Default window for aggregation (seconds).
            enable_alerts: Whether to evaluate alert rules.
        """
        self.window_size = window_size
        self.aggregation_window = aggregation_window
        self.enable_alerts = enable_alerts

        self._lock = threading.RLock()
        self._requests: deque[RequestMetrics] = deque(maxlen=window_size)
        self._alerts: List[Alert] = []
        self._alert_rules: List[AlertRule] = []
        self._alert_cooldowns: Dict[str, float] = {}
        self._alert_callbacks: List[Callable[[Alert], None]] = []
        self._request_counter = 0

        # Global aggregates (cumulative)
        self._cumulative: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"requests": 0, "successes": 0, "errors": 0, "tokens": 0, "cost": 0.0}
        )

        # Per-provider, per-model latency history for percentiles
        self._latency_histories: Dict[str, List[float]] = defaultdict(list)
        self._max_latency_history = 10000

    # ── Recording ─────────────────────────────────────────────────

    def start_request(
        self,
        provider: str,
        model: str,
        strategy: str = "unknown",
        tags: Optional[Dict[str, str]] = None,
    ) -> RequestMetrics:
        """Start tracking a new request.

        Args:
            provider: Provider name.
            model: Model name.
            strategy: Routing strategy used.
            tags: Optional metadata tags.

        Returns:
            RequestMetrics object to be completed later.
        """
        self._request_counter += 1
        request_id = f"req_{self._request_counter}_{int(time.time() * 1000)}"

        return RequestMetrics(
            request_id=request_id,
            provider=provider,
            model=model,
            start_time=time.time(),
            strategy=strategy,
            tags=tags or {},
        )

    def complete_request(
        self,
        metrics: RequestMetrics,
        success: bool = True,
        latency_ms: Optional[float] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost: float = 0.0,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        cached: bool = False,
    ) -> None:
        """Complete a request with measured results.

        Args:
            metrics: The RequestMetrics from start_request.
            success: Whether the request succeeded.
            latency_ms: Measured latency.
            prompt_tokens: Prompt token count.
            completion_tokens: Completion token count.
            cost: Cost in dollars.
            error_type: Error classification.
            error_message: Error details.
            cached: Whether result was from cache.
        """
        metrics.end_time = time.time()
        if latency_ms is None:
            metrics.latency_ms = (metrics.end_time - metrics.start_time) * 1000
        else:
            metrics.latency_ms = latency_ms

        metrics.success = success
        metrics.prompt_tokens = prompt_tokens
        metrics.completion_tokens = completion_tokens
        metrics.total_tokens = prompt_tokens + completion_tokens
        metrics.cost = cost
        metrics.error_type = error_type
        metrics.error_message = error_message
        metrics.cached = cached

        with self._lock:
            self._requests.append(metrics)

            # Update cumulative
            key = f"{metrics.provider}:{metrics.model}"
            cum = self._cumulative[key]
            cum["requests"] += 1
            if success:
                cum["successes"] += 1
            else:
                cum["errors"] += 1
            cum["tokens"] += metrics.total_tokens
            cum["cost"] += metrics.cost

            # Track latency history for percentiles
            latency_key = key
            hist = self._latency_histories[latency_key]
            hist.append(metrics.latency_ms)
            if len(hist) > self._max_latency_history:
                self._latency_histories[latency_key] = hist[-self._max_latency_history:]

        # Evaluate alerts
        if self.enable_alerts:
            self._evaluate_alerts(metrics)

    # ── Querying ──────────────────────────────────────────────────

    def get_recent_requests(self, limit: int = 100) -> List[RequestMetrics]:
        """Get recent request metrics.

        Args:
            limit: Max number of recent requests.

        Returns:
            List of RequestMetrics.
        """
        with self._lock:
            return list(self._requests)[-limit:]

    def get_aggregation(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        window_seconds: Optional[float] = None,
    ) -> List[AggregatedMetrics]:
        """Get aggregated metrics for a time window.

        Args:
            provider: Filter by provider.
            model: Filter by model.
            window_seconds: Window size in seconds.

        Returns:
            List of AggregatedMetrics.
        """
        window = window_seconds or self.aggregation_window
        now = time.time()
        cutoff = now - window

        with self._lock:
            # Group by provider:model
            groups: Dict[str, List[RequestMetrics]] = defaultdict(list)
            for req in self._requests:
                if req.start_time < cutoff:
                    continue
                if provider and req.provider != provider:
                    continue
                if model and req.model != model:
                    continue
                key = f"{req.provider}:{req.model}"
                groups[key].append(req)

        results = []
        for key, reqs in groups.items():
            parts = key.split(":", 1)
            prov, mdl = parts[0], parts[1] if len(parts) > 1 else "unknown"

            latencies = sorted([r.latency_ms for r in reqs])

            agg = AggregatedMetrics(
                provider=prov,
                model=mdl,
                window_start=cutoff,
                window_end=now,
                request_count=len(reqs),
                success_count=sum(1 for r in reqs if r.success),
                error_count=sum(1 for r in reqs if not r.success),
                total_latency_ms=sum(r.latency_ms for r in reqs),
                min_latency_ms=latencies[0] if latencies else 0,
                max_latency_ms=latencies[-1] if latencies else 0,
                p50_latency_ms=self._percentile(latencies, 50),
                p95_latency_ms=self._percentile(latencies, 95),
                p99_latency_ms=self._percentile(latencies, 99),
                total_prompt_tokens=sum(r.prompt_tokens for r in reqs),
                total_completion_tokens=sum(r.completion_tokens for r in reqs),
                total_tokens=sum(r.total_tokens for r in reqs),
                total_cost=sum(r.cost for r in reqs),
                cached_count=sum(1 for r in reqs if r.cached),
            )
            results.append(agg)

        return results

    def get_global_stats(self) -> Dict[str, Any]:
        """Get global cumulative statistics.

        Returns:
            Dict with global metrics.
        """
        with self._lock:
            total_requests = sum(c["requests"] for c in self._cumulative.values())
            total_successes = sum(c["successes"] for c in self._cumulative.values())
            total_errors = sum(c["errors"] for c in self._cumulative.values())
            total_tokens = sum(c["tokens"] for c in self._cumulative.values())
            total_cost = sum(c["cost"] for c in self._cumulative.values())

            # Recent throughput (requests per second, last 60s)
            now = time.time()
            recent_60s = sum(
                1 for r in self._requests if r.start_time > now - 60
            )

            return {
                "total_requests": total_requests,
                "total_successes": total_successes,
                "total_errors": total_errors,
                "success_rate": (
                    total_successes / total_requests if total_requests > 0 else 0.0
                ),
                "total_tokens": total_tokens,
                "total_cost": round(total_cost, 6),
                "throughput_rps": round(recent_60s / 60, 2),
                "by_provider_model": dict(self._cumulative),
            }

    def get_latency_percentiles(
        self, provider: Optional[str] = None, model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get latency percentiles for specific provider/model.

        Args:
            provider: Filter by provider.
            model: Filter by model.

        Returns:
            Dict with p50, p95, p99 latencies.
        """
        result: Dict[str, Any] = {}

        with self._lock:
            for key, history in self._latency_histories.items():
                parts = key.split(":", 1)
                p, m = parts[0], parts[1] if len(parts) > 1 else "unknown"

                if provider and p != provider:
                    continue
                if model and m != model:
                    continue

                sorted_lat = sorted(history)
                result[key] = {
                    "count": len(sorted_lat),
                    "p50_ms": self._percentile(sorted_lat, 50),
                    "p95_ms": self._percentile(sorted_lat, 95),
                    "p99_ms": self._percentile(sorted_lat, 99),
                    "min_ms": sorted_lat[0] if sorted_lat else 0,
                    "max_ms": sorted_lat[-1] if sorted_lat else 0,
                    "avg_ms": sum(sorted_lat) / len(sorted_lat) if sorted_lat else 0,
                }

        return result

    # ── Alerts ────────────────────────────────────────────────────

    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add an alert rule.

        Args:
            rule: AlertRule to add.
        """
        self._alert_rules.append(rule)

    def on_alert(self, callback: Callable[[Alert], None]) -> None:
        """Register a callback for triggered alerts.

        Args:
            callback: Function to call with Alert object.
        """
        self._alert_callbacks.append(callback)

    def _evaluate_alerts(self, metrics: RequestMetrics) -> None:
        """Evaluate alert rules against a new metric.

        Args:
            metrics: The request metrics to check.
        """
        now = time.time()
        with self._lock:
            for rule in self._alert_rules:
                # Check cooldown
                last_fired = self._alert_cooldowns.get(rule.name, 0)
                if now - last_fired < rule.cooldown_seconds:
                    continue

                # Check provider/model filters
                if rule.providers and metrics.provider not in rule.providers:
                    continue
                if rule.models and metrics.model not in rule.models:
                    continue

                # Get metric value
                value = self._get_metric_value(rule, metrics)

                # Check condition
                triggered = False
                if rule.condition == "gt" and value > rule.threshold:
                    triggered = True
                elif rule.condition == "lt" and value < rule.threshold:
                    triggered = True
                elif rule.condition == "gte" and value >= rule.threshold:
                    triggered = True
                elif rule.condition == "lte" and value <= rule.threshold:
                    triggered = True

                if triggered:
                    alert = Alert(
                        rule_name=rule.name,
                        metric=rule.metric,
                        value=value,
                        threshold=rule.threshold,
                        condition=rule.condition,
                        provider=metrics.provider,
                        model=metrics.model,
                    )
                    self._alerts.append(alert)
                    self._alert_cooldowns[rule.name] = now

                    # Notify callbacks (outside lock to avoid deadlocks)
                    for cb in self._alert_callbacks:
                        try:
                            cb(alert)
                        except Exception:
                            pass

    def _get_metric_value(
        self, rule: AlertRule, metrics: RequestMetrics
    ) -> float:
        """Get the current value for a metric type within the rule's window.

        Args:
            rule: The alert rule.
            metrics: The triggering request.

        Returns:
            Current metric value.
        """
        window_seconds = rule.window_seconds
        cutoff = time.time() - window_seconds

        recent: List[RequestMetrics] = []
        for r in self._requests:
            if r.start_time >= cutoff:
                if (not rule.providers or r.provider in rule.providers) and \
                   (not rule.models or r.model in rule.models):
                    recent.append(r)

        if not recent:
            return 0.0

        if rule.metric == MetricType.LATENCY:
            return sum(r.latency_ms for r in recent) / len(recent)
        elif rule.metric == MetricType.SUCCESS_RATE:
            successes = sum(1 for r in recent if r.success)
            return successes / len(recent)
        elif rule.metric == MetricType.ERROR_RATE:
            errors = sum(1 for r in recent if not r.success)
            return errors / len(recent)
        elif rule.metric == MetricType.COST:
            return sum(r.cost for r in recent)
        elif rule.metric == MetricType.TOKEN_COUNT:
            return sum(r.total_tokens for r in recent)
        elif rule.metric == MetricType.THROUGHPUT:
            if len(recent) < 2:
                return 0.0
            duration = recent[-1].start_time - recent[0].start_time
            return len(recent) / max(duration, 0.001)
        return 0.0

    def get_alerts(self, limit: int = 50) -> List[Alert]:
        """Get recent alerts.

        Args:
            limit: Max alerts to return.

        Returns:
            List of Alert objects.
        """
        with self._lock:
            return list(self._alerts[-limit:])

    def clear_alerts(self) -> None:
        """Clear all alerts."""
        with self._lock:
            self._alerts.clear()

    # ── Export ────────────────────────────────────────────────────

    def export_json(self, filepath: Optional[str] = None) -> str:
        """Export metrics to JSON format.

        Args:
            filepath: Optional file path to save to.

        Returns:
            JSON string.
        """
        data = {
            "global": self.get_global_stats(),
            "aggregations": [
                {
                    "provider": a.provider,
                    "model": a.model,
                    "request_count": a.request_count,
                    "success_rate": a.success_rate,
                    "avg_latency_ms": a.avg_latency_ms,
                    "p95_latency_ms": a.p95_latency_ms,
                    "total_cost": a.total_cost,
                    "throughput": a.throughput,
                }
                for a in self.get_aggregation()
            ],
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }

        json_str = json.dumps(data, indent=2, default=str)

        if filepath:
            with open(filepath, "w") as f:
                f.write(json_str)

        return json_str

    # ── Reset ─────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._requests.clear()
            self._alerts.clear()
            self._cumulative.clear()
            self._latency_histories.clear()
            self._request_counter = 0

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _percentile(sorted_data: List[float], percentile: float) -> float:
        """Calculate percentile from sorted data.

        Args:
            sorted_data: Sorted list of values.
            percentile: Percentile (0-100).

        Returns:
            Percentile value.
        """
        if not sorted_data:
            return 0.0
        k = (len(sorted_data) - 1) * percentile / 100.0
        f = int(k)
        c = k - f
        if f + 1 < len(sorted_data):
            return sorted_data[f] + c * (sorted_data[f + 1] - sorted_data[f])
        return sorted_data[f]
