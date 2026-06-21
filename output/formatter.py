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
