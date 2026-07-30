"""
Performance benchmarking for LLM routing.

Measures:
  - Latency (end-to-end and per-provider)
  - Throughput (requests per second)
  - Cost efficiency
  - Success rates under load
  - Concurrent request handling
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    name: str = "benchmark"
    num_requests: int = 100
    concurrency: int = 10
    warmup_requests: int = 5
    timeout_per_request: float = 60.0
    providers: List[str] = field(default_factory=list)
    models: List[str] = field(default_factory=list)
    strategies: List[str] = field(default_factory=list)
    request_template: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LatencyStats:
    """Latency statistics for a set of requests."""

    min_ms: float = 0.0
    max_ms: float = 0.0
    mean_ms: float = 0.0
    median_ms: float = 0.0
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    stddev_ms: float = 0.0
    samples: List[float] = field(default_factory=list)

    @classmethod
    def from_samples(cls, samples: List[float]) -> LatencyStats:
        """Create LatencyStats from a list of measurements.

        Args:
            samples: List of latency values in milliseconds.

        Returns:
            LatencyStats instance.
        """
        if not samples:
            return cls()

        sorted_samples = sorted(samples)
        n = len(sorted_samples)

        return cls(
            min_ms=min(samples),
            max_ms=max(samples),
            mean_ms=statistics.mean(samples),
            median_ms=statistics.median(samples),
            p50_ms=sorted_samples[int(n * 0.50)] if n > 0 else 0,
            p90_ms=sorted_samples[int(n * 0.90)] if n > 1 else sorted_samples[0],
            p95_ms=sorted_samples[int(n * 0.95)] if n > 1 else sorted_samples[0],
            p99_ms=sorted_samples[int(n * 0.99)] if n > 1 else sorted_samples[0],
            stddev_ms=statistics.stdev(samples) if n > 1 else 0.0,
            samples=sorted_samples,
        )


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""

    config: BenchmarkConfig
    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    success_rate: float = 0.0
    throughput_rps: float = 0.0
    total_duration_s: float = 0.0
    latency: LatencyStats = field(default_factory=LatencyStats)
    latency_by_provider: Dict[str, LatencyStats] = field(default_factory=dict)
    cost_total: float = 0.0
    cost_per_request: float = 0.0
    tokens_total: int = 0
    tokens_per_request: float = 0.0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────
# Benchmark Runner
# ──────────────────────────────────────────────────────────────────


class BenchmarkRunner:
    """Runs performance benchmarks on LLM routing systems.

    Supports concurrent load testing, latency measurement,
    and comparative analysis across providers and strategies.
    """

    def __init__(
        self,
        completion_fn: Optional[Callable] = None,
        config: Optional[BenchmarkConfig] = None,
    ):
        """Initialize the benchmark runner.

        Args:
            completion_fn: Async function for LLM completion.
            config: Benchmark configuration.
        """
        self.completion_fn = completion_fn
        self.config = config or BenchmarkConfig()

    # ── Single Request ────────────────────────────────────────────

    async def _run_single_request(
        self,
        request_data: Dict[str, Any],
        request_id: int,
    ) -> Tuple[bool, float, float, int, Optional[str]]:
        """Run a single benchmark request.

        Args:
            request_data: Request payload.
            request_id: Request identifier.

        Returns:
            Tuple of (success, latency_ms, cost, tokens, error).
        """
        start = time.monotonic()
        try:
            if self.completion_fn:
                response = await asyncio.wait_for(
                    self.completion_fn(request_data),
                    timeout=self.config.timeout_per_request,
                )
                latency = (time.monotonic() - start) * 1000
                cost = response.get("cost", 0.0) if isinstance(response, dict) else 0.0
                tokens = response.get("tokens", 0) if isinstance(response, dict) else 0
                return True, latency, cost, tokens, None
            else:
                # Simulate (for testing without actual API)
                await asyncio.sleep(0.01)
                latency = (time.monotonic() - start) * 1000
                return True, latency, 0.0, 10, None

        except asyncio.TimeoutError:
            return False, self.config.timeout_per_request * 1000, 0.0, 0, "timeout"
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return False, latency, 0.0, 0, str(e)

    # ── Benchmark ─────────────────────────────────────────────────

    async def run(
        self,
        config: Optional[BenchmarkConfig] = None,
    ) -> BenchmarkResult:
        """Run a full benchmark.

        Args:
            config: Override configuration.

        Returns:
            BenchmarkResult with all measurements.
        """
        cfg = config or self.config
        result = BenchmarkResult(config=cfg)
        result.total_requests = cfg.num_requests

        all_latencies: List[float] = []
        latencies_by_provider: Dict[str, List[float]] = {}
        errors: List[Dict[str, Any]] = []
        total_cost = 0.0
        total_tokens = 0

        # Warmup
        if cfg.warmup_requests > 0:
            warmup_tasks = [
                self._run_single_request(cfg.request_template, -i)
                for i in range(cfg.warmup_requests)
            ]
            await asyncio.gather(*warmup_tasks)

        # Main benchmark
        start_time = time.monotonic()

        # Run with concurrency control
        semaphore = asyncio.Semaphore(cfg.concurrency)

        async def bounded_request(i: int) -> Tuple[bool, float, float, int, Optional[str]]:
            async with semaphore:
                return await self._run_single_request(cfg.request_template, i)

        tasks = [bounded_request(i) for i in range(cfg.num_requests)]
        results = await asyncio.gather(*tasks)

        total_duration = time.monotonic() - start_time

        # Analyze results
        for i, (success, latency, cost, tokens, error) in enumerate(results):
            if success:
                result.successful += 1
                all_latencies.append(latency)
                total_cost += cost
                total_tokens += tokens

                # Group by provider if available
                provider = cfg.request_template.get("provider", "default")
                if provider not in latencies_by_provider:
                    latencies_by_provider[provider] = []
                latencies_by_provider[provider].append(latency)
            else:
                result.failed += 1
                errors.append({
                    "request_id": i,
                    "error": error,
                    "latency_ms": latency,
                })

        # Compute stats
        result.success_rate = (
            result.successful / cfg.num_requests
            if cfg.num_requests > 0
            else 0.0
        )
        result.throughput_rps = (
            cfg.num_requests / total_duration if total_duration > 0 else 0.0
        )
        result.total_duration_s = total_duration
        result.latency = LatencyStats.from_samples(all_latencies)
        result.cost_total = total_cost
        result.cost_per_request = total_cost / max(result.successful, 1)
        result.tokens_total = total_tokens
        result.tokens_per_request = total_tokens / max(result.successful, 1)
        result.errors = errors

        # Per-provider stats
        result.latency_by_provider = {
            prov: LatencyStats.from_samples(lats)
            for prov, lats in latencies_by_provider.items()
        }

        return result

    # ── Comparative Benchmark ─────────────────────────────────────

    async def compare(
        self,
        configs: List[BenchmarkConfig],
    ) -> List[BenchmarkResult]:
        """Run comparative benchmarks across multiple configurations.

        Args:
            configs: List of benchmark configurations to compare.

        Returns:
            List of BenchmarkResults, one per config.
        """
        results = []
        for cfg in configs:
            result = await self.run(cfg)
            results.append(result)
        return results

    async def compare_strategies(
        self,
        strategies: List[str],
        base_config: Optional[BenchmarkConfig] = None,
    ) -> Dict[str, BenchmarkResult]:
        """Compare performance across routing strategies.

        Args:
            strategies: Strategy names to compare.
            base_config: Base benchmark config.

        Returns:
            Dict mapping strategy name to BenchmarkResult.
        """
        cfg = base_config or self.config
        results = {}

        for strategy in strategies:
            strat_cfg = BenchmarkConfig(
                name=f"{cfg.name}_{strategy}",
                num_requests=cfg.num_requests,
                concurrency=cfg.concurrency,
                warmup_requests=cfg.warmup_requests,
                timeout_per_request=cfg.timeout_per_request,
                providers=cfg.providers,
                models=cfg.models,
                strategies=[strategy],
                request_template={**cfg.request_template, "strategy": strategy},
            )
            results[strategy] = await self.run(strat_cfg)

        return results

    # ── Report ────────────────────────────────────────────────────

    def generate_report(self, result: BenchmarkResult) -> str:
        """Generate a human-readable benchmark report.

        Args:
            result: Benchmark result.

        Returns:
            Formatted report string.
        """
        lines = [
            "=" * 60,
            f"  Benchmark: {result.config.name}",
            "=" * 60,
            "",
            f"  Total Requests:   {result.total_requests}",
            f"  Successful:       {result.successful}",
            f"  Failed:           {result.failed}",
            f"  Success Rate:     {result.success_rate:.2%}",
            f"  Duration:         {result.total_duration_s:.2f}s",
            f"  Throughput:       {result.throughput_rps:.1f} req/s",
            "",
            "  Latency:",
            f"    Min:    {result.latency.min_ms:.1f}ms",
            f"    Mean:   {result.latency.mean_ms:.1f}ms",
            f"    Median: {result.latency.median_ms:.1f}ms",
            f"    P90:    {result.latency.p90_ms:.1f}ms",
            f"    P95:    {result.latency.p95_ms:.1f}ms",
            f"    P99:    {result.latency.p99_ms:.1f}ms",
            f"    Max:    {result.latency.max_ms:.1f}ms",
            f"    StdDev: {result.latency.stddev_ms:.1f}ms",
            "",
            "  Cost:",
            f"    Total:    ${result.cost_total:.4f}",
            f"    Per Req:  ${result.cost_per_request:.6f}",
            "",
            "  Tokens:",
            f"    Total:    {result.tokens_total}",
            f"    Per Req:  {result.tokens_per_request:.1f}",
        ]

        if result.latency_by_provider:
            lines.append("")
            lines.append("  Per-Provider Latency:")
            for prov, stats in result.latency_by_provider.items():
                lines.append(f"    {prov}:")
                lines.append(f"      Mean: {stats.mean_ms:.1f}ms | "
                             f"P95: {stats.p95_ms:.1f}ms | "
                             f"Count: {len(stats.samples)}")

        if result.errors:
            lines.append("")
            lines.append(f"  Errors ({len(result.errors)}):")
            error_types: Dict[str, int] = {}
            for e in result.errors[:5]:
                err = e.get("error", "unknown")
                error_types[err] = error_types.get(err, 0) + 1
            for err_type, count in error_types.items():
                lines.append(f"    {err_type}: {count}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def compare_report(
        self,
        results: List[BenchmarkResult],
    ) -> str:
        """Generate a comparative report.

        Args:
            results: Multiple benchmark results.

        Returns:
            Comparative report string.
        """
        lines = [
            "=" * 70,
            "  Comparative Benchmark Report",
            "=" * 70,
            "",
            f"{'Name':<20} {'Success':>8} {'Mean Lat':>10} {'P95 Lat':>10} {'RPS':>8}",
            "-" * 60,
        ]

        for res in results:
            lines.append(
                f"{res.config.name:<20} "
                f"{res.success_rate:>7.1%} "
                f"{res.latency.mean_ms:>9.0f}ms "
                f"{res.latency.p95_ms:>9.0f}ms "
                f"{res.throughput_rps:>7.1f}"
            )

        lines.append("-" * 60)
        lines.append("")
        return "\n".join(lines)

    def export_json(self, result: BenchmarkResult, filepath: Optional[str] = None) -> str:
        """Export benchmark results to JSON.

        Args:
            result: Benchmark result.
            filepath: Optional file path to save.

        Returns:
            JSON string.
        """
        data = {
            "config": {
                "name": result.config.name,
                "num_requests": result.config.num_requests,
                "concurrency": result.config.concurrency,
            },
            "results": {
                "total_requests": result.total_requests,
                "successful": result.successful,
                "failed": result.failed,
                "success_rate": result.success_rate,
                "throughput_rps": result.throughput_rps,
                "total_duration_s": result.total_duration_s,
                "latency": {
                    "min_ms": result.latency.min_ms,
                    "mean_ms": result.latency.mean_ms,
                    "median_ms": result.latency.median_ms,
                    "p90_ms": result.latency.p90_ms,
                    "p95_ms": result.latency.p95_ms,
                    "p99_ms": result.latency.p99_ms,
                    "max_ms": result.latency.max_ms,
                    "stddev_ms": result.latency.stddev_ms,
                },
                "cost_total": result.cost_total,
                "cost_per_request": result.cost_per_request,
                "tokens_total": result.tokens_total,
                "tokens_per_request": result.tokens_per_request,
                "errors_count": len(result.errors),
            },
        }

        json_str = json.dumps(data, indent=2)

        if filepath:
            with open(filepath, "w") as f:
                f.write(json_str)

        return json_str
