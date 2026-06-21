# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

"""
workflows/tool_use/evaluator.py

Scores a tool-use TaskResult against the task's expected_calls.

Unlike summarization, this is MOSTLY MECHANICAL — we compare what the
agent actually called against what we know a correct run should call, so
most of this doesn't need an LLM judge at all. This is a deliberate
design choice: mechanical scoring is fast, free, deterministic, and not
subject to judge bias (see workflows/summarization/evaluator.py's
anti-bias rule for why that bias matters in the summarization case).

Four dimensions, all 0-100:

    Tool Selection      - did it call the right tools, and avoid wrong/
                           irrelevant ones?
    Argument Correctness- were the arguments to each call correct?
    Task Completion     - did all REQUIRED calls happen by the end?
    Efficiency           - did it avoid redundant/repeated calls?

Plus Latency and Cost, same as summarization, normalized the same way.
"""

from typing import Dict

from core.types import TaskResult, ScoreResult
from workflows.base import Evaluator
from workflows.tool_use.tasks import ToolUseTask, ExpectedCall

DEFAULT_WEIGHTS = {
    "tool_selection": 0.25,
    "argument_correctness": 0.25,
    "task_completion": 0.25,
    "efficiency": 0.10,
    "latency": 0.075,
    "cost": 0.075,
}


class ToolUseEvaluator(Evaluator):
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or DEFAULT_WEIGHTS
        self._latency_range = (0.0, 0.0)
        self._cost_range = (0.0, 0.0)

    def score_all(self, task_results_with_meta: list, task: ToolUseTask) -> list:
        """Same batching pattern as SummarizationEvaluator.score_all() —
        latency/cost are normalized relative to the whole batch."""
        latencies = [tr.latency_ms for _, _, tr in task_results_with_meta]
        costs = [tr.cost_usd for _, _, tr in task_results_with_meta]
        self._latency_range = (min(latencies), max(latencies))
        self._cost_range = (min(costs), max(costs))

        scored = []
        for framework, model_id, task_result in task_results_with_meta:
            score_result = self.score(task_result, task)
            scored.append((framework, model_id, task_result, score_result))
        return scored

    def score(self, task_result: TaskResult, task_input: ToolUseTask) -> ScoreResult:
        task = task_input
        actual_calls = task_result.extra.get("tool_calls", [])

        tool_selection = self._score_tool_selection(actual_calls, task.expected_calls)
        arg_correctness = self._score_argument_correctness(actual_calls, task.expected_calls)
        completion = self._score_task_completion(actual_calls, task.expected_calls)
        efficiency = self._score_efficiency(actual_calls, task.expected_calls)

        latency_score = self.normalize_inverted(task_result.latency_ms, *self._latency_range)
        cost_score = self.normalize_inverted(task_result.cost_usd, *self._cost_range)

        dimensions = {
            "tool_selection": tool_selection,
            "argument_correctness": arg_correctness,
            "task_completion": completion,
            "efficiency": efficiency,
            "latency": {"score": latency_score, "reason": f"{task_result.latency_ms/1000:.1f}s"},
            "cost": {"score": cost_score, "reason": f"${task_result.cost_usd:.4f}"},
        }

        composite = sum(
            dimensions[name]["score"] * weight
            for name, weight in self.weights.items()
        )

        return ScoreResult(dimensions=dimensions, composite_score=round(composite, 1))

    def _score_tool_selection(self, actual_calls: list, expected_calls: list) -> dict:
        """
        Did the agent call the tools it was supposed to, and avoid calling
        tools it wasn't? Scored as: (correct tools called) / (expected
        tools + any wrong tools called), so both missing AND extra wrong
        calls hurt the score.
        """
        actual_tool_names = [c["tool_name"] for c in actual_calls]
        expected_tool_names = [c.tool for c in expected_calls]

        correct = sum(1 for t in expected_tool_names if t in actual_tool_names)
        wrong = sum(1 for t in actual_tool_names if t not in expected_tool_names)
        total_expected = len(expected_tool_names)

        if total_expected == 0:
            score = 100.0 if wrong == 0 else 0.0
        else:
            score = max(0.0, round((correct - wrong) / total_expected * 100, 1))

        reason = f"{correct}/{total_expected} expected tools called"
        if wrong:
            reason += f", {wrong} unexpected call(s)"

        return {"score": score, "reason": reason}

    def _score_argument_correctness(self, actual_calls: list, expected_calls: list) -> dict:
        """
        For each expected call that WAS made, check how many of its
        expected arguments matched. This only checks calls that happened —
        missing calls are already penalized by tool_selection/completion.
        """
        by_tool = {c["tool_name"]: c["arguments"] for c in actual_calls}

        total_args_checked = 0
        total_args_correct = 0

        for expected in expected_calls:
            actual_args = by_tool.get(expected.tool)
            if actual_args is None:
                continue  # call never happened — not scored here
            for key, expected_value in expected.args.items():
                total_args_checked += 1
                actual_value = actual_args.get(key)
                if self._values_roughly_match(actual_value, expected_value):
                    total_args_correct += 1

        if total_args_checked == 0:
            return {"score": 0.0, "reason": "No matching tool calls to check arguments on"}

        score = round(total_args_correct / total_args_checked * 100, 1)
        return {"score": score, "reason": f"{total_args_correct}/{total_args_checked} arguments matched"}

    def _score_task_completion(self, actual_calls: list, expected_calls: list) -> dict:
        """Did every REQUIRED expected call happen, regardless of extras?"""
        actual_tool_names = set(c["tool_name"] for c in actual_calls)
        required = [c for c in expected_calls if c.required]

        if not required:
            return {"score": 100.0, "reason": "No required calls for this task"}

        completed = sum(1 for c in required if c.tool in actual_tool_names)
        score = round(completed / len(required) * 100, 1)
        return {"score": score, "reason": f"{completed}/{len(required)} required steps completed"}

    def _score_efficiency(self, actual_calls: list, expected_calls: list) -> dict:
        """
        Penalize redundant calls — e.g. calling the same tool with the
        same arguments more than once. A perfectly efficient run makes
        exactly as many calls as there are expected calls.
        """
        expected_count = len(expected_calls)
        actual_count = len(actual_calls)

        if expected_count == 0:
            return {"score": 100.0 if actual_count == 0 else 50.0, "reason": "No expected calls"}

        extra_calls = max(0, actual_count - expected_count)
        score = max(0.0, round(100 - (extra_calls / expected_count * 50), 1))
        reason = f"{actual_count} calls made, {expected_count} expected"
        return {"score": score, "reason": reason}

    def _values_roughly_match(self, actual, expected) -> bool:
        """Loose comparison: case-insensitive for strings, exact for
        everything else. Real-world agent output often differs in casing
        or whitespace without being meaningfully wrong."""
        if isinstance(actual, str) and isinstance(expected, str):
            return actual.strip().lower() == expected.strip().lower()
        return actual == expected
