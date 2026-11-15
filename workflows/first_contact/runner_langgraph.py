# SPDX-License-Identifier: Apache-2.0
"""
workflows/first_contact/runner_langgraph.py

Stateful sequential graph orchestration (LangGraph pattern) for First Contact:
  Node 1 ("technical_node"):  drafts Technical/Scientific pillars (Comms, Bio, Infra, Ecology)
  Node 2 ("human_node"):      drafts Governance/Social/Economic pillars (Gov, Psych, Econ)
  Node 3 ("synthesize_node"): combines both state nodes into the unified 7-pillar plan + JSON

Because each node runs sequentially in a state graph, wall-clock latency is the
sum of all sequential node calls.
"""

from typing import TypedDict

from core.model_router import ModelRouter
from core.types import TaskResult
from workflows.base import Workflow
from workflows.first_contact.task import FirstContactTask, TASK_INSTRUCTIONS

NODE1_PROMPT = """You are Node 1 (Scientific, Biosecurity & Defense Node) in a LangGraph state machine.
In 200 words max, draft concrete technical mechanisms and budget/staffing splits for these 4 pillars (out of $10.0B and 1,000 total staff):
1. communication_cryptography (prime/math universal language, passive listen-first signal processing, encryption)
2. biosafety_xenobiology (BSL-5+ airlock quarantine, extremophile/prion pathogen screening, seed vault)
3. infrastructure_defense (modular shielded habitats, EMP-hardened redundant cyber/comms, defensive posture)
4. planetary_ecology (artifact decontamination, protected zones, biodiversity monitoring)

QUESTION:
{question}"""

NODE2_PROMPT = """You are Node 2 (Governance, Psychology & Economics Node) in a LangGraph state machine.
In 150 words max, draft concrete mechanisms and budget/staffing splits for the remaining 3 pillars (so all 7 pillars sum to <= $10.0B and exactly 1,000 staff):
5. governance_diplomacy (UN multilateral treaty, Earth Consensus Protocol, game-theory negotiation, spokesperson chain of command)
6. psychology_sociology (graduated disclosure, cosmic anxiety counseling, crew isolation screening)
7. economics_resources (market reserves, IP/trade rules, closed-loop nuclear/solar energy, food, water)

NODE 1 OUTPUT:
{node1_draft}"""

NODE3_PROMPT = """You are Node 3 (Final State Assembler) in a LangGraph state machine.
Combine Node 1 and Node 2 below into the final First Contact Preparedness Plan.
Keep the total response under 550 words, starting with the ```json ... ``` block followed by the 7-pillar summary.

{instructions}

NODE 1 STATE:
{node1_draft}

NODE 2 STATE:
{node2_draft}"""


class FirstContactGraphState(TypedDict):
    model_id: str
    node1_draft: str
    node2_draft: str
    final_output: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float


class FirstContactLangGraphRunner(Workflow):
    def __init__(self, router: ModelRouter):
        self.router = router

    def run(self, model_id: str, task_input: FirstContactTask) -> TaskResult:
        task = task_input
        state: FirstContactGraphState = {
            "model_id": model_id,
            "node1_draft": "",
            "node2_draft": "",
            "final_output": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": 0.0,
            "cost_usd": 0.0,
        }

        r1 = self.router.call(model_id, NODE1_PROMPT.format(question=task.question))
        state["node1_draft"] = r1.text
        for k, v in (("input_tokens", r1.input_tokens), ("output_tokens", r1.output_tokens), ("latency_ms", r1.latency_ms), ("cost_usd", r1.cost_usd)):
            state[k] += v  # type: ignore

        r2 = self.router.call(model_id, NODE2_PROMPT.format(node1_draft=state["node1_draft"]))
        state["node2_draft"] = r2.text
        for k, v in (("input_tokens", r2.input_tokens), ("output_tokens", r2.output_tokens), ("latency_ms", r2.latency_ms), ("cost_usd", r2.cost_usd)):
            state[k] += v  # type: ignore

        r3 = self.router.call(
            model_id,
            NODE3_PROMPT.format(
                instructions=TASK_INSTRUCTIONS,
                node1_draft=state["node1_draft"],
                node2_draft=state["node2_draft"],
            ),
        )
        state["final_output"] = r3.text
        for k, v in (("input_tokens", r3.input_tokens), ("output_tokens", r3.output_tokens), ("latency_ms", r3.latency_ms), ("cost_usd", r3.cost_usd)):
            state[k] += v  # type: ignore

        return TaskResult(
            output=state["final_output"],
            input_tokens=state["input_tokens"],
            output_tokens=state["output_tokens"],
            latency_ms=state["latency_ms"],
            cost_usd=state["cost_usd"],
            extra={"total_calls": 3, "sequential_hops": 3},
        )
