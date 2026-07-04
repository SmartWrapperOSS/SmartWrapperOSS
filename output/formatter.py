# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

"""
output/formatter.py

Renders results as a terminal comparison table and saves results.json.

KEY DESIGN POINT: this file does NOT know about "Quality" or "Coverage"
or "Tool Selection" by name. It just reads whatever dimension names are
present in each result's ScoreResult.dimensions dict, and builds columns
from those. This is what lets a brand new workflow (with its own, totally
different dimension names) show up correctly in the table with ZERO
changes to this file.
"""

import json
from typing import List
from core.types import EvalResult


def print_table(results: List[EvalResult], show_reasons: bool = True):
    if not results:
        print("No results to display.")
        return

    # Every result in a single run comes from the same workflow, so they
    # all share the same dimension names — just read them from the first.
    dimension_names = list(results[0].score_result.dimensions.keys())

    header = _build_header(dimension_names)
    divider = "─" * len(header)

    print()
    print(divider)
    print(header)
    print(divider)

    for r in results:
        print(_build_row(r, dimension_names))
        if show_reasons:
            _print_reasons(r, dimension_names)

    print(divider)
    best = results[0]  # results are pre-sorted by composite_score, highest first
    print(f"\nBest: {best.framework} + {best.model_id} (composite score: {best.composite_score})")
    print()


def _build_header(dimension_names: List[str]) -> str:
    columns = [f"{'Framework':<12}", f"{'Model':<22}"]
    for name in dimension_names:
        columns.append(f"{name.replace('_', ' ').title():>14}")
    columns.append(f"{'Score':>8}")
    return " ".join(columns)


def _build_row(result: EvalResult, dimension_names: List[str]) -> str:
    columns = [f"{result.framework:<12}", f"{result.model_id:<22}"]
    for name in dimension_names:
        dim = result.score_result.dimensions[name]
        value = dim["score"]
        # Latency/cost dimensions store something more readable in `reason`
        # (e.g. "1.8s" or "$0.018") — show that instead of the raw score.
        display = dim.get("reason") if name in ("latency", "cost") else f"{value}"
        columns.append(f"{display:>14}")
    columns.append(f"{result.composite_score:>8}")
    return " ".join(columns)


def _print_reasons(result: EvalResult, dimension_names: List[str]):
    for name in dimension_names:
        dim = result.score_result.dimensions[name]
        reason = dim.get("reason", "")
        if reason and name not in ("latency", "cost"):
            label = name.replace("_", " ").title()
            print(f"  {label}: {reason}")
    print()


def save_json(results: List[EvalResult], path: str):
    data = []
    for r in results:
        data.append({
            "framework": r.framework,
            "model_id": r.model_id,
            "composite_score": r.composite_score,
            "dimensions": r.score_result.dimensions,
            "output": r.task_result.output,
            "latency_ms": r.task_result.latency_ms,
            "cost_usd": r.task_result.cost_usd,
        })

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to {path}")


# --- Aggregated (multi-run) rendering ---------------------------------------
#
# Same design rule as print_table above: no dimension names are hardcoded.
# Whatever absolute-scale dimensions exist in the aggregates become columns,
# rendered as "mean ±std". Latency and cost get raw-value columns
# (median / p95 ms, mean $) because their 0-100 scores are batch-normalized
# and cannot be averaged across runs — see core/stats.py.

from core.stats import ComboAggregate  # noqa: E402


def print_aggregate_table(aggs: "List[ComboAggregate]"):
    if not aggs:
        print("No results to display.")
        return

    n_values = sorted({a.n for a in aggs})
    n_desc = f"n={n_values[0]}" if len(n_values) == 1 else f"n={n_values[0]}..{n_values[-1]}"
    dimension_names = list(aggs[0].dimensions.keys())

    columns = [f"{'Framework':<12}", f"{'Model':<22}"]
    for name in dimension_names:
        columns.append(f"{name.replace('_', ' ').title():>16}")
    columns += [f"{'Lat med/p95':>16}", f"{'Cost mean':>11}", f"{'Composite':>16}"]
    header = " ".join(columns)
    divider = "─" * len(header)

    print()
    print(f"Aggregated over repeated runs ({n_desc}); score cells are mean ±std dev; 95% CI on composite.")
    print(divider)
    print(header)
    print(divider)

    for a in aggs:
        row = [f"{a.framework:<12}", f"{a.model_id:<22}"]
        for name in dimension_names:
            m = a.dimensions[name]
            row.append(f"{f'{m.mean:.1f} ±{m.std:.1f}':>16}")
        row.append(f"{f'{a.latency_ms_median/1000:.1f}s/{a.latency_ms_p95/1000:.1f}s':>16}")
        row.append(f"{f'${a.cost_usd_mean:.4f}':>11}")
        c = a.composite
        row.append(f"{f'{c.mean:.1f} ±{c.std:.1f}':>16}")
        print(" ".join(row))
        print(f"    composite 95% CI: [{c.ci_low:.1f}, {c.ci_high:.1f}]"
              f"   range: [{c.min:.1f}, {c.max:.1f}]   n={a.n}")

    print(divider)

    best = aggs[0]
    print(f"\nBest (by mean composite): {best.framework} + {best.model_id} "
          f"({best.composite.mean:.1f} ±{best.composite.std:.1f}, n={best.n})")

    # Honest significance note: CI overlap between the top two rows.
    # Non-overlap is a conservative "this difference looks real" signal;
    # overlap means "cannot distinguish at this n" — NOT "they're equal".
    if len(aggs) >= 2 and best.n >= 2 and aggs[1].n >= 2:
        runner_up = aggs[1]
        if best.composite.overlaps(runner_up.composite):
            print(f"Note: 95% CIs of the top two overlap — this ranking is NOT "
                  f"statistically distinguishable at n={best.n}. Add runs to tighten.")
        else:
            print(f"Note: 95% CIs of the top two do not overlap — the lead of "
                  f"{best.framework}+{best.model_id} over {runner_up.framework}+"
                  f"{runner_up.model_id} holds at this sample size.")
    print()


def save_aggregate_json(aggs: "List[ComboAggregate]", path: str):
    data = []
    for a in aggs:
        data.append({
            "framework": a.framework,
            "model_id": a.model_id,
            "n_runs": a.n,
            "composite": vars(a.composite),
            "dimensions": {k: vars(v) for k, v in a.dimensions.items()},
            "latency_ms": {"median": a.latency_ms_median, "p95": a.latency_ms_p95},
            "cost_usd": {"mean": a.cost_usd_mean, "total": a.cost_usd_total},
        })
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Aggregate results saved to {path}")
