# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

"""
workflows/summarization/runner_autogen.py

Summarizes a document using a simulated AutoGen-style two-agent pattern:
  Agent 1 ("Summarizer") drafts a summary.
  Agent 2 ("Reviewer")   critiques and improves it.

This mirrors AutoGen's conversational multi-agent design: two roles
passing a message back and forth, rather than an explicit state graph
(compare with runner_langgraph.py, which does the same two-step idea but
via explicit state transitions instead).
"""

from typing import List

from core.types import TaskResult
from core.model_router import ModelRouter
from ingestion.document_loader import Chunk
from workflows.base import Workflow

SUMMARIZE_PROMPT = """You are a document summarization assistant.
Summarize the following document clearly and concisely, capturing all key points.

DOCUMENT:
{document}

Provide a structured summary with the most important information."""

REVIEW_PROMPT = """You are a summary quality reviewer.
Review the following summary against the original document excerpts.
Improve it if anything important is missing or unclear. Return only the
final improved summary, with no preamble.

ORIGINAL DOCUMENT EXCERPTS:
{document}

DRAFT SUMMARY:
{draft}"""


class SummarizationAutoGenRunner(Workflow):
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
        built-in text above if not overridden, so existing behavior is
        unchanged unless a config explicitly opts in.

        Each template must contain the same {placeholder} names as the
        defaults above ({document} for summarize_prompt; {document} and
        {draft} for review_prompt), since those are filled in at run time.
        """
        self.router = router
        self.summarize_prompt = summarize_prompt
        self.review_prompt = review_prompt

    def run(self, model_id: str, task_input: List[Chunk]) -> TaskResult:
        """`task_input` is the list of document Chunks to summarize."""
        chunks = task_input
        document_text = "\n\n".join(c.text for c in chunks)

        # --- Agent 1: Summarizer ---
        draft_prompt = self.summarize_prompt.format(document=document_text)
        draft = self.router.call(model_id, draft_prompt)

        # --- Agent 2: Reviewer (a second pass over the draft) ---
        review_prompt = self.review_prompt.format(document=document_text, draft=draft.text)
        final = self.router.call(model_id, review_prompt)

        # Combine usage from both agent turns into one TaskResult
        return TaskResult(
            output=final.text,
            input_tokens=draft.input_tokens + final.input_tokens,
            output_tokens=draft.output_tokens + final.output_tokens,
            latency_ms=draft.latency_ms + final.latency_ms,
            cost_usd=draft.cost_usd + final.cost_usd,
        )
