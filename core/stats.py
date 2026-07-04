# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

"""
core/stats.py

Aggregates repeated runs of the same (framework, model) combination into
mean / std dev / 95% confidence intervals, so the comparison table can
show "84.2 ±1.9 (n=5)" instead of a single-sample number.

DESIGN NOTES
- No numpy/scipy dependency on purpose. Everything here is computable
  with the stdlib `statistics` module plus a small lookup table of
  two-tailed Student's t critical values. The t-table matters because at
  realistic run counts (n = 3..10, given API costs) the normal-approx
  1.96 meaningfully understates the interval width.
- Latency and cost are aggregated on their RAW values (ms, USD), never
  on their normalized 0-100 dimension scores. Those scores are
  normalized *within one batch* (see Evaluator.normalize_inverted), so
  they are not comparable across sessions and must not be averaged.
  Judge/mechanical dimension scores (quality, tool_selection, ...) ARE
  absolute 0-100 values, so those aggregate fine.
- Latency uses median/p95 rather than mean because latency is
  long-tailed: one slow API call would otherwise skew the whole cell.
"""

from dataclasses import dataclass, field
from statistics import mean, stdev, median
from typing import Any, Dict, List, Optional

# Dimensions whose per-run "score" is batch-normalized and therefore
# must NOT be averaged across runs. Their raw values are aggregated
# separately (latency_ms, cost_usd on TaskResult).
BATCH_NORMALIZED_DIMENSIONS = ("latency", "cost")

# Two-tailed 95% critical values of Student's t, indexed by degrees of
# freedom (n - 1). Beyond df=30 the normal value 1.96 is close enough.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def t95(df: int) -> float:
    if df < 1:
        return float("inf")
    return _T95.get(df, 1.960)


def percentile(values: List[float], p: float) -> float:
    """Nearest-rank percentile (p in 0..100). Good enough at small n."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, round(p / 100.0 * (len(ordered) - 1))))
    return ordered[k]


@dataclass
class MetricAggregate:
    """Summary statistics for one metric over n runs."""
    n: int
    mean: float
    std: float        # sample std dev; 0.0 when n == 1
    ci_low: float     # 95% CI (t-distribution); equals mean when n == 1
    ci_high: float
    min: float
    max: float

    @property
    def ci_half_width(self) -> float:
        return (self.ci_high - self.ci_low) / 2.0

    def overlaps(self, other: "MetricAggregate") -> bool:
        """True if the two 95% CIs overlap. A non-overlap is a simple,
        conservative signal that a difference is real; an overlap does
        NOT prove there's no difference — don't overclaim either way."""
        return not (self.ci_high < other.ci_low or other.ci_high < self.ci_low)


def summarize(values: List[float]) -> MetricAggregate:
    n = len(values)
    if n == 0:
        return MetricAggregate(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    m = mean(values)
    if n == 1:
        return MetricAggregate(1, m, 0.0, m, m, m, m)
    s = stdev(values)  # sample std dev (ddof=1)
    half = t95(n - 1) * s / (n ** 0.5)
    return MetricAggregate(
        n=n, mean=round(m, 2), std=round(s, 2),
        ci_low=round(m - half, 2), ci_high=round(m + half, 2),
        min=min(values), max=max(values),
    )


@dataclass
class ComboAggregate:
    """One row of the aggregated comparison table: everything we know
    about a (framework, model) combination over n runs."""
    framework: str
    model_id: str
    n: int
    # dimension name -> MetricAggregate, for absolute-scale dimensions only
    dimensions: Dict[str, MetricAggregate] = field(default_factory=dict)
    composite: Optional[MetricAggregate] = None
    latency_ms_median: float = 0.0
    latency_ms_p95: float = 0.0
    cost_usd_mean: float = 0.0
    cost_usd_total: float = 0.0


def aggregate_runs(records: List[Dict[str, Any]]) -> List[ComboAggregate]:
    """
    records: raw run dicts as persisted by core/runstore.py. Each must
    contain: framework, model_id, dimensions, composite_score,
    latency_ms, cost_usd.

    Returns one ComboAggregate per (framework, model_id), sorted by mean
    composite score, highest first.
    """
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for rec in records:
        groups.setdefault((rec["framework"], rec["model_id"]), []).append(rec)

    aggs: List[ComboAggregate] = []
    for (fw, model_id), recs in groups.items():
        agg = ComboAggregate(framework=fw, model_id=model_id, n=len(recs))

        dim_names = [
            name for name in recs[0]["dimensions"]
            if name not in BATCH_NORMALIZED_DIMENSIONS
        ]
        for name in dim_names:
            scores = [float(r["dimensions"][name]["score"]) for r in recs
                      if name in r["dimensions"]]
            agg.dimensions[name] = summarize(scores)

        agg.composite = summarize([float(r["composite_score"]) for r in recs])

        latencies = [float(r["latency_ms"]) for r in recs]
        costs = [float(r["cost_usd"]) for r in recs]
        agg.latency_ms_median = round(median(latencies), 1)
        agg.latency_ms_p95 = round(percentile(latencies, 95), 1)
        agg.cost_usd_mean = round(mean(costs), 6)
        agg.cost_usd_total = round(sum(costs), 6)

        aggs.append(agg)

    aggs.sort(key=lambda a: a.composite.mean if a.composite else 0.0, reverse=True)
    return aggs
