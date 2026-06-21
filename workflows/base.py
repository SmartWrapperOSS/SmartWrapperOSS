# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

"""
workflows/base.py

THIS IS THE MOST IMPORTANT FILE IN THE PROJECT TO UNDERSTAND.

SmartWrapperOSS supports multiple kinds of tasks (summarization, tool-use,
and more in the future). Each task type is called a "workflow". For a new
workflow to plug into the existing CLI, comparison table, and JSON output
with zero changes to those files, every workflow must follow the same
two-part contract defined here:

  1. A Workflow  — runs ONE (framework, model) combination on a task and
                    returns a TaskResult (see core/types.py).
  2. An Evaluator — scores ONE TaskResult and returns a ScoreResult.

Why split "running" from "scoring" into two separate classes?
Because they change independently. You might want to improve how tool-use
agents are scored without touching how they're run, or vice versa. Keeping
them separate also makes each class small enough to read in one sitting.

HOW TO ADD A NEW WORKFLOW (e.g. "RAG / Question-Answering"):
  1. Create workflows/rag/runner_autogen.py and runner_langgraph.py,
     each implementing the Workflow interface below.
  2. Create workflows/rag/evaluator.py implementing the Evaluator interface.
  3. Register both in main.py's WORKFLOW_REGISTRY (one dict entry).
  4. Done. The CLI, comparison table, and JSON output need NO changes —
     they only ever talk to the Workflow/Evaluator interfaces, never to
     anything summarization- or tool-use-specific.
"""

from abc import ABC, abstractmethod
from core.types import TaskResult, ScoreResult


class Workflow(ABC):
    """
    Runs a task using one orchestration framework (e.g. AutoGen or
    LangGraph) and one model, and returns what happened as a TaskResult.

    Concrete examples:
        - SummarizationAutoGenRunner: summarizes a document via AutoGen
        - ToolUseLangGraphRunner: runs a tool-calling agent via LangGraph
    """

    @abstractmethod
    def run(self, model_id: str, task_input) -> TaskResult:
        """
        Execute the task for `model_id`.

        `task_input` is intentionally untyped here (it differs per
        workflow — a list of document Chunks for summarization, a
        ToolUseTask for tool-use). Each concrete Workflow documents what
        it expects in its own file.
        """
        raise NotImplementedError


class Evaluator(ABC):
    """
    Scores a single TaskResult and returns a ScoreResult.

    Concrete examples:
        - SummarizationEvaluator: scores Quality/Coverage/Conciseness via
          an LLM judge, plus Latency/Cost.
        - ToolUseEvaluator: scores ToolSelection/ArgumentCorrectness/
          TaskCompletion/Efficiency mechanically, plus Latency/Cost.
    """

    @abstractmethod
    def score(self, task_result: TaskResult, task_input) -> ScoreResult:
        """
        Score `task_result`. `task_input` is passed through so the
        evaluator can compare against the original task (e.g. the source
        document for summarization, or the expected tool calls for
        tool-use).
        """
        raise NotImplementedError

    def normalize_inverted(self, value: float, min_val: float, max_val: float) -> float:
        """
        Shared helper: turns a "lower is better" raw number (like latency
        or cost) into a 0-100 score where higher is better, so it can be
        combined with judge scores on the same scale.

        Every Evaluator needs this for latency/cost, so it lives here
        once instead of being copy-pasted into each one.
        """
        if max_val == min_val:
            return 100.0
        return round(100 - ((value - min_val) / (max_val - min_val) * 100), 1)
