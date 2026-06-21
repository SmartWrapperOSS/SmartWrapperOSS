# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

"""
workflows/summarization/evaluator.py

Scores a summary on three subjective dimensions using an LLM as a judge,
plus latency and cost (objective, no judge needed):

    Quality      - factual accuracy, coherence, no hallucinations
    Coverage     - how much of the source document's key info is captured
    Conciseness  - tight and clear, without losing meaning

ANTI-BIAS RULE: the judge model is never the same model that produced the
summary being judged. Models tend to rate their own output more favorably,
so we always pick a different model to judge — see _pick_judge() below.
"""

import json
from typing import Dict

from core.types import TaskResult, ScoreResult
from core.model_router import ModelRouter
from workflows.base import Evaluator

JUDGE_PROMPTS = {
    "quality": """You are evaluating the factual quality of a document summary.

SOURCE DOCUMENT (excerpts):
{source}

SUMMARY:
{summary}

Score the summary on QUALITY (factual accuracy, coherence, no hallucinations) from 0 to 100.
Return ONLY valid JSON, nothing else: {{"score": <integer 0-100>, "reason": "<max 15 words>"}}""",

    "coverage": """You are evaluating how completely a summary covers the source document.

SOURCE DOCUMENT (excerpts):
{source}

SUMMARY:
{summary}

Score the summary on COVERAGE (what percentage of key points are captured) from 0 to 100.
Return ONLY valid JSON, nothing else: {{"score": <integer 0-100>, "reason": "<max 15 words>"}}""",

    "conciseness": """You are evaluating the conciseness of a document summary.

SOURCE DOCUMENT (excerpts):
{source}

SUMMARY:
{summary}

Score the summary on CONCISENESS (tight and clear without losing meaning) from 0 to 100.
Return ONLY valid JSON, nothing else: {{"score": <integer 0-100>, "reason": "<max 15 words>"}}""",
}

DEFAULT_WEIGHTS = {
    "quality": 0.35,
    "coverage": 0.30,
    "conciseness": 0.15,
    "latency": 0.10,
    "cost": 0.10,
}


class SummarizationEvaluator(Evaluator):
    def __init__(self, router: ModelRouter, judge_model_id: str, weights: Dict[str, float] = None):
        self.router = router
        self.judge_model_id = judge_model_id
        self.weights = weights or DEFAULT_WEIGHTS

        # Filled in by score_all() before any individual score() calls, so
        # latency/cost can be normalized relative to the whole batch.
        self._latency_range = (0.0, 0.0)
        self._cost_range = (0.0, 0.0)

    def score_all(self, task_results_with_meta: list, source_text: str) -> list:
        """
        Score a whole batch of (framework, model_id, TaskResult) tuples at
        once. Batching matters here because latency/cost normalization
        needs to know the min/max across ALL results, not just one.
        """
        latencies = [tr.latency_ms for _, _, tr in task_results_with_meta]
        costs = [tr.cost_usd for _, _, tr in task_results_with_meta]
        self._latency_range = (min(latencies), max(latencies))
        self._cost_range = (min(costs), max(costs))

        scored = []
        for framework, model_id, task_result in task_results_with_meta:
            score_result = self.score(task_result, source_text, evaluated_model_id=model_id)
            scored.append((framework, model_id, task_result, score_result))
        return scored

    def score(self, task_result: TaskResult, task_input, evaluated_model_id: str = None) -> ScoreResult:
        """`task_input` here is the source document text used as reference."""
        source_text = task_input
        summary = task_result.output

        judge_id = self._pick_judge(evaluated_model_id)

        dimensions = {}
        for dimension_name in ["quality", "coverage", "conciseness"]:
            score, reason = self._ask_judge(judge_id, dimension_name, source_text, summary)
            dimensions[dimension_name] = {"score": score, "reason": reason}

        latency_score = self.normalize_inverted(task_result.latency_ms, *self._latency_range)
        cost_score = self.normalize_inverted(task_result.cost_usd, *self._cost_range)
        dimensions["latency"] = {"score": latency_score, "reason": f"{task_result.latency_ms/1000:.1f}s"}
        dimensions["cost"] = {"score": cost_score, "reason": f"${task_result.cost_usd:.4f}"}

        composite = sum(
            dimensions[name]["score"] * weight
            for name, weight in self.weights.items()
        )

        return ScoreResult(dimensions=dimensions, composite_score=round(composite, 1))

    def _ask_judge(self, judge_model_id: str, dimension: str, source: str, summary: str):
        prompt = JUDGE_PROMPTS[dimension].format(source=source, summary=summary)
        try:
            response = self.router.call(judge_model_id, prompt)
            data = self._parse_judge_json(response.text)
            return int(data.get("score", 50)), data.get("reason", "")
        except Exception as e:
            # If the judge call fails or returns unparseable output, fall
            # back to a neutral score rather than crashing the whole run.
            return 50, f"Judge error: {str(e)[:40]}"

    def _parse_judge_json(self, raw: str) -> dict:
        """Judges sometimes wrap JSON in markdown code fences - strip those first."""
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)

    def _pick_judge(self, evaluated_model_id: str) -> str:
        """Pick a judge model that is NOT the model being evaluated (anti-bias rule)."""
        if evaluated_model_id != self.judge_model_id:
            return self.judge_model_id
        # The configured judge IS the model under evaluation this round —
        # fall back to any other available model instead.
        for candidate in self.router.configs.keys():
            if candidate != evaluated_model_id:
                return candidate
        return self.judge_model_id  # only one model configured — no choice
