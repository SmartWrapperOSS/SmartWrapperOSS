# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

"""
workflows/guidance_extraction/runner_autogen.py

Extracts guidance with a simulated AutoGen-style two-agent conversation:
  Agent 1 ("Analyst")  reads every document and drafts the full JSON answer.
  Agent 2 ("Reviewer") checks the draft against the sources, removes anything
                       that isn't company guidance, and returns the final JSON.

Compare with runner_langgraph.py, which decomposes the same job into
per-document extraction nodes plus a separate comparison node.
"""

from core.model_router import ModelRouter
from core.types import TaskResult
from workflows.base import Workflow
from workflows.guidance_extraction.task import (
    OUTPUT_FORMAT, RULES, GuidanceTask, documents_block, metrics_block,
)

ANALYST_PROMPT = """You are a financial analyst extracting company guidance from earnings calls.

METRICS
{metrics}

{rules}

DOCUMENTS (earliest first)
{documents}

{output_format}"""

REVIEWER_PROMPT = """You are reviewing another analyst's guidance extraction for accuracy.
Check every item against the source documents. Remove anything that is not
forward-looking guidance stated by the company (reported results, consensus,
industry forecasts), fix wrong numbers, units, periods, or change types, and
add anything that was missed.

METRICS
{metrics}

{rules}

DOCUMENTS (earliest first)
{documents}

DRAFT
{draft}

{output_format}"""


class GuidanceAutoGenRunner(Workflow):
    def __init__(self, router: ModelRouter):
        self.router = router

    def run(self, model_id: str, task_input: GuidanceTask) -> TaskResult:
        task = task_input
        common = dict(metrics=metrics_block(task), rules=RULES,
                      documents=documents_block(task), output_format=OUTPUT_FORMAT)

        draft = self.router.call(model_id, ANALYST_PROMPT.format(**common))
        final = self.router.call(model_id, REVIEWER_PROMPT.format(draft=draft.text, **common))

        steps = [draft, final]
        return TaskResult(
            output=final.text,
            input_tokens=sum(s.input_tokens for s in steps),
            output_tokens=sum(s.output_tokens for s in steps),
            latency_ms=sum(s.latency_ms for s in steps),
            cost_usd=sum(s.cost_usd for s in steps),
            extra={"draft": draft.text},
        )
