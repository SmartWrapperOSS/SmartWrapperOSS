# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

"""
core/types.py

Shared data shapes used by every workflow.

Why this file exists:
Summarization and Tool-Use are very different tasks, but they both need to
flow through the SAME pipeline: Model Router -> Comparison Table -> JSON
output. To make that possible, every workflow must produce results in a
common shape, even though what's INSIDE that shape differs per workflow.

Think of it like a shipping container: the contents differ (furniture vs.
electronics), but the container size is standard so any truck/ship/crane
can move it. TaskResult and ScoreResult are our shipping containers.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class TaskResult:
    """
    The raw output of running ONE (framework, model) combination on a task.

    This is intentionally generic:
    - For summarization, `output` is the summary text.
    - For tool-use, `output` is a description of what happened (e.g. final
      answer text), and the actual tool calls live in `extra["tool_calls"]`.

    Every workflow MUST fill in: output, input_tokens, output_tokens,
    latency_ms, cost_usd. The `extra` dict is where workflow-specific data
    (like tool call traces) lives, without forcing every other workflow to
    carry unused fields.
    """
    output: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoreResult:
    """
    The scored output of evaluating ONE TaskResult.

    `dimensions` holds whatever metrics the active workflow's Evaluator
    decided to score, e.g.:
        {
            "quality":  {"score": 91, "reason": "Accurate, no hallucinations"},
            "coverage": {"score": 88, "reason": "Missed Q3 guidance"},
        }
    or, for tool-use:
        {
            "tool_selection":  {"score": 100, "reason": "Correct tool chosen"},
            "argument_correctness": {"score": 80, "reason": "Wrong date format"},
        }

    composite_score is always a single 0-100 number computed from the
    dimensions above, using the active workflow's configured weights — this
    is what lets the comparison table sort results, regardless of workflow.
    """
    dimensions: Dict[str, Dict[str, Any]]
    composite_score: float = 0.0


@dataclass
class EvalResult:
    """
    One row in the final comparison table: a (framework, model) combination,
    its raw output, and its scores. This is what formatter.py renders.
    """
    framework: str
    model_id: str
    task_result: TaskResult
    score_result: ScoreResult

    @property
    def composite_score(self) -> float:
        return self.score_result.composite_score
