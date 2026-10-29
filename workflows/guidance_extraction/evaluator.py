# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

"""
workflows/guidance_extraction/evaluator.py

Scores a guidance-extraction answer against the task's answer key. Fully
MECHANICAL: the YAML fixes every expected number, so no LLM judge is
involved and the same output always gets the same score.

Weighted dimensions (default weights, configurable):

    schema_validity       10%  the answer parses as JSON with the right shape;
                               every item uses a listed metric, a valid
                               period, numeric-or-null bounds, and a known
                               change type. For graph runners that assemble
                               JSON in code, scaled by the share of node
                               outputs that parsed
    extraction_accuracy   30%  for each expected guidance item: was it found
                               (same source, metric, period), and do low and
                               high match within the metric's tolerance?
    change_detection      30%  for each expected change: detected, right
                               type, right "from" bounds, right "to" bounds.
                               Each change the key doesn't contain costs as
                               much as missing a real one (e.g. claiming Q3
                               revenue guidance was "withdrawn" after Q3
                               simply ended).
    grounding             15%  share of reported guidance items that are real
                               guidance: not a trap, not invented, not a
                               duplicate
    latency, cost       7.5% each, batch-normalized like other workflows

Reported with weight 0 (its own pass/fail column; mean over runs = pass rate):

    critical_invariants  FAILS if any reported guidance item is a
                         hallucination: a statement the task marks as a trap
                         (a reported actual, analyst consensus) or a
                         metric/period the company never guided. One
                         hallucinated number in a financial product is a
                         broken answer, however good the rest is, so a high
                         average must not hide it.
"""

from typing import Dict, List, Optional, Tuple

from core.types import ScoreResult, TaskResult
from workflows.base import Evaluator
from workflows.guidance_extraction.task import (
    CHANGE_TYPES, PERIOD_RE, GuidanceTask, norm_metric, norm_period, parse_output,
)

DEFAULT_WEIGHTS = {
    "schema_validity": 0.10,
    "extraction_accuracy": 0.30,
    "change_detection": 0.30,
    "grounding": 0.15,
    "critical_invariants": 0.0,
    "latency": 0.075,
    "cost": 0.075,
}

_NUM = (int, float)


def _is_bound(v) -> bool:
    return v is None or (isinstance(v, _NUM) and not isinstance(v, bool))


def _close(a: Optional[float], b: Optional[float], tol: float) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol + 1e-9


def _bounds_match(actual, expected, tol: float) -> bool:
    """`expected` is None (no guidance) or {low, high}. An actual of None,
    or a dict whose bounds are both null, both count as 'no guidance'."""
    if isinstance(actual, dict) and actual.get("low") is None and actual.get("high") is None:
        actual = None
    if expected is None or actual is None:
        return expected is None and actual is None
    return (_close(actual.get("low"), expected["low"], tol)
            and _close(actual.get("high"), expected["high"], tol))


class GuidanceExtractionEvaluator(Evaluator):
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or DEFAULT_WEIGHTS
        self._latency_range = (0.0, 0.0)
        self._cost_range = (0.0, 0.0)

    def score_all(self, task_results_with_meta: list, task: GuidanceTask) -> list:
        latencies = [tr.latency_ms for _, _, tr in task_results_with_meta]
        costs = [tr.cost_usd for _, _, tr in task_results_with_meta]
        self._latency_range = (min(latencies), max(latencies))
        self._cost_range = (min(costs), max(costs))
        return [(fw, m, tr, self.score(tr, task)) for fw, m, tr in task_results_with_meta]

    def score(self, task_result: TaskResult, task_input: GuidanceTask) -> ScoreResult:
        task = task_input
        parsed = parse_output(task_result.output)

        schema, guidance, changes = self._schema(parsed, task)
        # Runners that assemble the final JSON in code (runner_langgraph.py)
        # report how many model steps returned unparseable output; scale the
        # schema score by it so code-side assembly can't earn format points
        # the model didn't.
        total = task_result.extra.get("nodes_total", 0)
        unparsed = task_result.extra.get("nodes_unparsed", 0)
        if total and unparsed:
            schema = {"score": round(schema["score"] * (1 - unparsed / total), 1),
                      "reason": f"{unparsed}/{total} graph node outputs unparseable"
                      + ("" if schema["reason"].startswith("Valid JSON") else "; " + schema["reason"])}
        extraction, grounding, critical = self._guidance(guidance, task)
        change = self._changes(changes, task)

        dims = {
            "schema_validity": schema,
            "extraction_accuracy": extraction,
            "change_detection": change,
            "grounding": grounding,
            "critical_invariants": critical,
            "latency": {"score": self.normalize_inverted(task_result.latency_ms, *self._latency_range),
                        "reason": f"{task_result.latency_ms / 1000:.1f}s"},
            "cost": {"score": self.normalize_inverted(task_result.cost_usd, *self._cost_range),
                     "reason": f"${task_result.cost_usd:.4f}"},
        }
        composite = sum(dims[name]["score"] * w for name, w in self.weights.items() if name in dims)
        return ScoreResult(dimensions=dims, composite_score=round(composite, 1))

    # --- schema_validity ----------------------------------------------------

    def _schema(self, parsed, task: GuidanceTask) -> Tuple[dict, List[dict], List[dict]]:
        """Returns the dimension plus the VALID, normalized guidance and
        change items. Invalid items are penalized here only and excluded
        from every other dimension, so one formatting slip isn't punished
        three times over."""
        if parsed is None:
            return {"score": 0.0, "reason": "Output is not parseable JSON"}, [], []

        problems = []
        shape_ok = isinstance(parsed.get("guidance"), list) and isinstance(parsed.get("changes"), list)
        if not shape_ok:
            problems.append("missing 'guidance' or 'changes' list")
        raw_g = parsed.get("guidance") if isinstance(parsed.get("guidance"), list) else []
        raw_c = parsed.get("changes") if isinstance(parsed.get("changes"), list) else []

        guidance = []
        for item in raw_g:
            why = self._invalid_guidance(item, task)
            if why:
                problems.append(f"guidance item {why}")
            else:
                guidance.append({"source": item["source"], "metric": norm_metric(item["metric"]),
                                 "period": norm_period(item["period"]),
                                 "low": item.get("low"), "high": item.get("high")})

        changes = []
        for item in raw_c:
            why = self._invalid_change(item, task)
            if why:
                problems.append(f"change item {why}")
            else:
                changes.append({"metric": norm_metric(item["metric"]), "period": norm_period(item["period"]),
                                "type": item["type"].strip().lower(),
                                "from": item.get("from"), "to": item.get("to")})

        frac_g = len(guidance) / len(raw_g) if raw_g else 1.0
        frac_c = len(changes) / len(raw_c) if raw_c else 1.0
        score = round((1 + shape_ok + frac_g + frac_c) / 4 * 100, 1)
        reason = "Valid JSON, all items well-formed" if not problems else \
            f"{len(problems)} problem(s): " + "; ".join(problems[:3]) + ("; ..." if len(problems) > 3 else "")
        return {"score": score, "reason": reason}, guidance, changes

    def _invalid_guidance(self, item, task: GuidanceTask) -> Optional[str]:
        if not isinstance(item, dict):
            return "is not an object"
        if item.get("source") not in task.doc_ids:
            return f"has unknown source {item.get('source')!r}"
        if norm_metric(item.get("metric")) not in task.metrics:
            return f"uses unlisted metric {item.get('metric')!r}"
        if not PERIOD_RE.match(norm_period(item.get("period"))):
            return f"has malformed period {item.get('period')!r}"
        low, high = item.get("low"), item.get("high")
        if not (_is_bound(low) and _is_bound(high)) or (low is None and high is None):
            return f"has non-numeric or empty bounds ({low!r}, {high!r})"
        if low is not None and high is not None and low > high:
            return f"has low > high ({low}, {high})"
        return None

    def _invalid_change(self, item, task: GuidanceTask) -> Optional[str]:
        if not isinstance(item, dict):
            return "is not an object"
        if norm_metric(item.get("metric")) not in task.metrics:
            return f"uses unlisted metric {item.get('metric')!r}"
        if not PERIOD_RE.match(norm_period(item.get("period"))):
            return f"has malformed period {item.get('period')!r}"
        if str(item.get("type", "")).strip().lower() not in CHANGE_TYPES:
            return f"has unknown type {item.get('type')!r}"
        for side in ("from", "to"):
            b = item.get(side)
            if b is not None and not (isinstance(b, dict) and _is_bound(b.get("low")) and _is_bound(b.get("high"))):
                return f"has malformed '{side}' bounds"
        return None

    # --- extraction_accuracy, grounding, critical_invariants ----------------

    def _guidance(self, items: List[dict], task: GuidanceTask) -> Tuple[dict, dict, dict]:
        unused = list(range(len(items)))
        earned, total = 0, 3 * len(task.expected_guidance)
        misses, trap_hits, invented, duplicates = [], [], [], 0

        def value_hits(item, exp, tol):
            return _close(item["low"], exp.low, tol) + _close(item["high"], exp.high, tol)

        for exp in task.expected_guidance:
            tol = task.metrics[exp.metric].tol
            cands = [i for i in unused if (items[i]["source"], items[i]["metric"], items[i]["period"])
                     == (exp.source, exp.metric, exp.period)]
            if not cands:
                misses.append(f"{exp.source} {exp.metric} {exp.period}")
                continue
            best = max(cands, key=lambda i: value_hits(items[i], exp, tol))
            unused.remove(best)
            hits = value_hits(items[best], exp, tol)
            earned += 1 + hits
            if hits < 2:
                trap = self._trap_for(items[best], task)
                misses.append(f"{exp.source} {exp.metric} {exp.period} wrong value"
                              + (f" ({trap})" if trap else ""))
                if trap:
                    trap_hits.append(trap)

        # Whatever is left over matched no expected item.
        for i in unused:
            item = items[i]
            trap = self._trap_for(item, task)
            if trap:
                trap_hits.append(trap)
                continue
            exp = next((e for e in task.expected_guidance
                        if (e.source, e.metric, e.period) == (item["source"], item["metric"], item["period"])), None)
            if exp and value_hits(item, exp, task.metrics[exp.metric].tol) == 2:
                duplicates += 1
            else:
                invented.append(f"{item['source']} {item['metric']} {item['period']} "
                                f"({item['low']}, {item['high']})")

        extraction = {
            "score": round(earned / total * 100, 1) if total else 100.0,
            "reason": f"{earned}/{total} item fields correct"
                      + (f"; missed or wrong: {', '.join(misses[:4])}" if misses else ""),
        }

        bad = len(trap_hits) + len(invented) + duplicates
        if items:
            g_score = round((len(items) - bad) / len(items) * 100, 1)
            g_reason = f"{len(items) - bad}/{len(items)} reported items are real guidance"
            if duplicates:
                g_reason += f"; {duplicates} duplicate(s)"
        else:
            g_score, g_reason = 0.0, "No guidance items reported"
        grounding = {"score": max(0.0, g_score), "reason": g_reason}

        hallucinations = trap_hits + invented
        critical = {
            "score": 0.0 if hallucinations else 100.0,
            "reason": "FAIL: " + "; ".join(hallucinations[:3]) + ("; ..." if len(hallucinations) > 3 else "")
            if hallucinations else "PASS: no hallucinated guidance",
        }
        return extraction, grounding, critical

    @staticmethod
    def _trap_for(item: dict, task: GuidanceTask) -> Optional[str]:
        """The trap this item fell for, if any. A trap with a value only
        matches when the reported number equals it, so the real $4.1B FY
        guidance isn't mistaken for the $4.0B consensus trap that shares
        its metric and period."""
        for t in task.traps:
            if (t.source, t.metric, t.period) != (item["source"], item["metric"], item["period"]):
                continue
            tol = task.metrics[t.metric].tol
            if t.value is None or _close(item["low"], t.value, tol) or _close(item["high"], t.value, tol):
                return t.description
        return None

    # --- change_detection ---------------------------------------------------

    def _changes(self, items: List[dict], task: GuidanceTask) -> dict:
        unused = list(range(len(items)))
        earned, total = 0, 4 * len(task.expected_changes)
        notes = []

        for exp in task.expected_changes:
            tol = task.metrics[exp.metric].tol
            cands = [i for i in unused if (items[i]["metric"], items[i]["period"]) == (exp.metric, exp.period)]
            if not cands:
                notes.append(f"missed {exp.metric} {exp.period} {exp.type}")
                continue

            def pts(i):
                c = items[i]
                return ((c["type"] in exp.accept) + _bounds_match(c["from"], exp.from_, tol)
                        + _bounds_match(c["to"], exp.to, tol))

            best = max(cands, key=pts)
            unused.remove(best)
            earned += 1 + pts(best)
            if items[best]["type"] not in exp.accept:
                notes.append(f"{exp.metric} {exp.period} called {items[best]['type']}, expected {exp.type}")
            elif pts(best) < 3:
                notes.append(f"{exp.metric} {exp.period} wrong from/to values")

        false_changes = [f"{items[i]['metric']} {items[i]['period']} {items[i]['type']}" for i in unused]
        score = max(0.0, round((earned - 4 * len(false_changes)) / total * 100, 1)) if total else 100.0
        reason = f"{earned}/{total} change points"
        if false_changes:
            reason += f", {len(false_changes)} false change(s): {', '.join(false_changes[:3])}"
        if notes:
            reason += f"; {', '.join(notes[:3])}"
        return {"score": score, "reason": reason}
