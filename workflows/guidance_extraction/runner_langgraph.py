# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

"""
workflows/guidance_extraction/runner_langgraph.py

Extracts guidance with a LangGraph-style explicit state graph:

    extract_node (one per document) -> compare_node -> assemble_node -> done

Each extraction node sees ONE document, so it can't confuse which call said
what. The comparison node then sees both extracted lists plus the source
text (it needs the text to notice guidance that was withdrawn, which by
definition leaves nothing to extract). assemble_node is plain code: it
merges the JSON pieces into the final answer and makes no model call.

Because assembly is code, the final answer is always valid JSON even if a
node's model output wasn't. To keep that from handing this runner free
schema_validity points, every node whose output fails to parse is counted
in extra["nodes_unparsed"], and the evaluator scales schema_validity by it.

Same task as runner_autogen.py, decomposed differently, which is exactly
the kind of orchestration choice this project exists to measure.
"""

import json
from typing import List, TypedDict

from core.model_router import ModelRouter
from core.types import TaskResult
from workflows.base import Workflow
from workflows.guidance_extraction.task import (
    RULES, GuidanceTask, documents_block, metrics_block, parse_output,
)

EXTRACT_PROMPT = """You are a financial analyst extracting company guidance from ONE earnings call.

METRICS
{metrics}

{rules}

DOCUMENT
{document}

Respond with ONLY a JSON object, no prose and no Markdown fences:
{{"guidance": [{{"source": "{doc_id}", "metric": "<metric>", "period": "<period>",
                "low": <number or null>, "high": <number or null>}}]}}"""

COMPARE_PROMPT = """You are a financial analyst comparing company guidance across earnings calls.

METRICS
{metrics}

{rules}

DOCUMENTS (earliest first)
{documents}

GUIDANCE ALREADY EXTRACTED FROM EACH DOCUMENT
{extracted}

Report how guidance changed from the earliest to the latest document.
Respond with ONLY a JSON object, no prose and no Markdown fences:
{{"changes": [{{"metric": "<metric>", "period": "<period>", "type": "<change type>",
               "from": {{"low": <number or null>, "high": <number or null>}} or null,
               "to": {{"low": <number or null>, "high": <number or null>}} or null}}]}}"""


class GraphState(TypedDict):
    model_id: str
    guidance: List[dict]
    changes: List[dict]
    raw_steps: List[str]
    nodes_unparsed: int
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float


class GuidanceLangGraphRunner(Workflow):
    def __init__(self, router: ModelRouter):
        self.router = router

    def run(self, model_id: str, task_input: GuidanceTask) -> TaskResult:
        task = task_input
        state: GraphState = {"model_id": model_id, "guidance": [], "changes": [], "raw_steps": [], "nodes_unparsed": 0,
                             "input_tokens": 0, "output_tokens": 0, "latency_ms": 0.0, "cost_usd": 0.0}

        for doc in task.documents:
            state = self._extract_node(state, task, doc.id)
        state = self._compare_node(state, task)
        output = self._assemble_node(state)

        return TaskResult(
            output=output,
            input_tokens=state["input_tokens"],
            output_tokens=state["output_tokens"],
            latency_ms=state["latency_ms"],
            cost_usd=state["cost_usd"],
            extra={"node_outputs": state["raw_steps"], "nodes_total": len(state["raw_steps"]),
                   "nodes_unparsed": state["nodes_unparsed"]},
        )

    def _call(self, state: GraphState, prompt: str) -> str:
        r = self.router.call(state["model_id"], prompt)
        state["raw_steps"].append(r.text)
        state["input_tokens"] += r.input_tokens
        state["output_tokens"] += r.output_tokens
        state["latency_ms"] += r.latency_ms
        state["cost_usd"] += r.cost_usd
        return r.text

    def _extract_node(self, state: GraphState, task: GuidanceTask, doc_id: str) -> GraphState:
        """One node per document: extract only that call's guidance."""
        text = self._call(state, EXTRACT_PROMPT.format(
            metrics=metrics_block(task), rules=RULES,
            document=documents_block(task, [doc_id]), doc_id=doc_id))
        parsed = parse_output(text)
        items = (parsed or {}).get("guidance")
        if isinstance(items, list):
            state["guidance"].extend(items)
        else:
            state["nodes_unparsed"] += 1
        return state

    def _compare_node(self, state: GraphState, task: GuidanceTask) -> GraphState:
        text = self._call(state, COMPARE_PROMPT.format(
            metrics=metrics_block(task), rules=RULES, documents=documents_block(task),
            extracted=json.dumps(state["guidance"], indent=2)))
        parsed = parse_output(text)
        changes = (parsed or {}).get("changes")
        if isinstance(changes, list):
            state["changes"] = changes
        else:
            state["nodes_unparsed"] += 1
        return state

    def _assemble_node(self, state: GraphState) -> str:
        """Deterministic merge, no model call."""
        return json.dumps({"guidance": state["guidance"], "changes": state["changes"]}, indent=2)
