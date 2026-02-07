# utils_stats.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import math


@dataclass
class LatencyStats:
    n: int
    mean_ms: float
    stdev_ms: float
    min_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float


def _percentile(sorted_vals: List[float], p: float) -> float:
    """
    Compute percentile using linear interpolation between closest ranks.
    sorted_vals must be sorted ascending.
    p in [0, 100].
    """
    if not sorted_vals:
        return float("nan")
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]

    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def compute_latency_stats(latencies_ms: List[float]) -> LatencyStats:
    vals = [float(x) for x in latencies_ms if x is not None]
    vals.sort()
    n = len(vals)
    if n == 0:
        return LatencyStats(0, float("nan"), float("nan"), float("nan"),
                            float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))

    mean = sum(vals) / n
    if n >= 2:
        var = sum((x - mean) ** 2 for x in vals) / (n - 1)
        stdev = math.sqrt(var)
    else:
        stdev = 0.0

    return LatencyStats(
        n=n,
        mean_ms=mean,
        stdev_ms=stdev,
        min_ms=vals[0],
        p50_ms=_percentile(vals, 50),
        p90_ms=_percentile(vals, 90),
        p95_ms=_percentile(vals, 95),
        p99_ms=_percentile(vals, 99),
        max_ms=vals[-1],
    )


def stats_to_dict(stats: LatencyStats) -> Dict[str, Any]:
    return {
        "n": stats.n,
        "mean_ms": stats.mean_ms,
        "stdev_ms": stats.stdev_ms,
        "min_ms": stats.min_ms,
        "p50_ms": stats.p50_ms,
        "p90_ms": stats.p90_ms,
        "p95_ms": stats.p95_ms,
        "p99_ms": stats.p99_ms,
        "max_ms": stats.max_ms,
    }