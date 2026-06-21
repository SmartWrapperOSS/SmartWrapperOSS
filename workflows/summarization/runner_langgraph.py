# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

"""
workflows/summarization/runner_langgraph.py

Summarizes a document using a LangGraph-style explicit state graph:

    summarize_node -> review_node -> done

The state (draft summary, token counts, etc.) is passed explicitly between
nodes as a typed dict, which is LangGraph's signature style — contrast
with runner_autogen.py, which achieves a similar two-step result through
conversational agent messages instead. Same task, two different
orchestration philosophies — which is exactly what this project is built
to compare.
"""

from typing import List, TypedDict

from core.types import TaskResult
from core.model_router import ModelRouter
from ingestion.document_loader import Chunk
from workflows.base import Workflow

SUMMARIZE_PROMPT = """Summarize the following document, capturing all key points clearly.

DOCUMENT:
{document}"""

REVIEW_PROMPT = """Review and refine this summary for accuracy and completeness.
Return only the final improved summary, with no preamble.

ORIGINAL:
{document}

SUMMARY:
{draft}"""


class GraphState(TypedDict):
    """The state passed between graph nodes — LangGraph's core concept."""
    document_text: str
    draft_summary: str
    final_summary: str
    total_input_tokens: int
    total_output_tokens: int
    total_latency_ms: float
    total_cost_usd: float
    model_id: str
    summarize_prompt: str
    review_prompt: str


class SummarizationLangGraphRunner(Workflow):
    def __init__(
        self,
        router: ModelRouter,
        summarize_prompt: str = SUMMARIZE_PROMPT,
        review_prompt: str = REVIEW_PROMPT,
    ):
        """
        `summarize_prompt` and `review_prompt` can be overridden — e.g. from
        config/config.yaml's `prompts:` section — to test how prompt wording
        affects scores, without touching this file. Both default to the
        built-in text above if not overridden.

        Each template must contain the same {placeholder} names as the
        defaults above ({document} for summarize_prompt; {document} and
        {draft} for review_prompt), since those are filled in at run time.
        """
        self.router = router
        self.summarize_prompt = summarize_prompt
        self.review_prompt = review_prompt

    def run(self, model_id: str, task_input: List[Chunk]) -> TaskResult:
        """`task_input` is the list of document Chunks to summarize."""
        document_text = "\n\n".join(c.text for c in task_input)

        state: GraphState = {
            "document_text": document_text,
            "draft_summary": "",
            "final_summary": "",
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_latency_ms": 0.0,
            "total_cost_usd": 0.0,
            "model_id": model_id,
            "summarize_prompt": self.summarize_prompt,
            "review_prompt": self.review_prompt,
        }

        state = self._summarize_node(state)
        state = self._review_node(state)

        return TaskResult(
            output=state["final_summary"],
            input_tokens=state["total_input_tokens"],
            output_tokens=state["total_output_tokens"],
            latency_ms=state["total_latency_ms"],
            cost_usd=state["total_cost_usd"],
        )

    def _summarize_node(self, state: GraphState) -> GraphState:
        """Node 1: generate a draft summary from the document."""
        prompt = state["summarize_prompt"].format(document=state["document_text"])
        response = self.router.call(state["model_id"], prompt)

        state["draft_summary"] = response.text
        state["total_input_tokens"] += response.input_tokens
        state["total_output_tokens"] += response.output_tokens
        state["total_latency_ms"] += response.latency_ms
        state["total_cost_usd"] += response.cost_usd
        return state

    def _review_node(self, state: GraphState) -> GraphState:
        """Node 2: refine the draft into a final summary."""
        prompt = state["review_prompt"].format(
            document=state["document_text"],
            draft=state["draft_summary"],
        )
        response = self.router.call(state["model_id"], prompt)

        state["final_summary"] = response.text
        state["total_input_tokens"] += response.input_tokens
        state["total_output_tokens"] += response.output_tokens
        state["total_latency_ms"] += response.latency_ms
        state["total_cost_usd"] += response.cost_usd
        return state
