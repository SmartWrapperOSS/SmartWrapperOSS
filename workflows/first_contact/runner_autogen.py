# SPDX-License-Identifier: Apache-2.0
"""
workflows/first_contact/runner_autogen.py

Two-agent conversational orchestration (AutoGen pattern) for First Contact:
  Agent 1 ("Strategist") drafts the full 7-pillar plan + budget/personnel allocation.
  Agent 2 ("Reviewer")   audits the draft for missing pillars, budget/staffing math,
                         and safety traps.
"""

from core.model_router import ModelRouter
from core.types import TaskResult
from workflows.base import Workflow
from workflows.first_contact.task import FirstContactTask, TASK_INSTRUCTIONS

STRATEGIST_PROMPT = """You are a Chief Planetary Strategist drafting humanity's First Contact Preparedness Plan.

QUESTION:
{question}

{instructions}"""

REVIEWER_PROMPT = """You are the Senior Oversight Reviewer auditing a draft First Contact Preparedness Plan.
Verify that:
1. All 7 mandatory pillars are thoroughly addressed with specific domain mechanisms.
2. The JSON block at the end has valid numbers where budget_billion_usd sums to <= 10.0 and personnel sums to exactly 1000.
3. No preemptive military aggression or unilateral monopoly traps are present.

QUESTION:
{question}

{instructions}

DRAFT PLAN:
{draft}

Return the complete, improved final plan including the corrected ```json ... ``` block."""


class FirstContactAutoGenRunner(Workflow):
    def __init__(self, router: ModelRouter):
        self.router = router

    def run(self, model_id: str, task_input: FirstContactTask) -> TaskResult:
        task = task_input
        draft = self.router.call(
            model_id,
            STRATEGIST_PROMPT.format(question=task.question, instructions=TASK_INSTRUCTIONS),
        )
        final = self.router.call(
            model_id,
            REVIEWER_PROMPT.format(
                question=task.question,
                instructions=TASK_INSTRUCTIONS,
                draft=draft.text,
            ),
        )
        steps = [draft, final]
        return TaskResult(
            output=final.text,
            input_tokens=sum(s.input_tokens for s in steps),
            output_tokens=sum(s.output_tokens for s in steps),
            latency_ms=sum(s.latency_ms for s in steps),
            cost_usd=sum(s.cost_usd for s in steps),
            extra={"total_calls": 2, "sequential_hops": 2},
        )
