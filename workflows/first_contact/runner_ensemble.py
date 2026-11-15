# SPDX-License-Identifier: Apache-2.0
"""
workflows/first_contact/runner_ensemble.py

Parallel Swarm + Consensus Synthesizer (Ask Swarm / Ember Swarm pattern)
for the First Contact with Aliens benchmark:
  Phase 1 ("Swarm"):       N expert personas (from the 100-persona Ask Swarm pool:
                           Architect, Biologist, Psychologist, Economist, Diplomat,
                           Cryptographer, Ethicist, Military Strategist, etc.)
                           analyze the problem concurrently via ThreadPoolExecutor.
  Phase 2 ("Synthesizer"): 1 Consensus Synthesizer merges all N expert perspectives
                           into the 7 mandatory pillars and enforces the $10.0B /
                           1,000-person constraints.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from core.model_router import ModelResponse, ModelRouter
from core.types import TaskResult
from workflows.base import Workflow
from workflows.first_contact.task import FirstContactTask, TASK_INSTRUCTIONS

ASK_SWARM_EXPERTS = [
    "Architect", "Biologist", "Psychologist", "Economist", "Diplomat",
    "Cryptographer", "Ethicist", "Military Strategist", "Neuroscientist",
    "Futurist", "Space Psychologist", "Xenobiologist", "Environmental Scientist",
    "Mechanical Engineer", "Urban Planner", "Nutritionist", "Materials Scientist",
    "Sociologist", "Doctor", "Electrical Engineer", "Historian", "Philosopher",
    "Data Scientist", "Security Expert", "Logistics Manager", "Teacher", "Artist",
    "Farmer", "Chemist", "Geologist", "Astronomer", "Communications Expert",
    "Lawyer", "Energy Engineer", "Robotics Engineer", "Software Developer",
    "Anthropologist", "Geneticist", "Game Designer", "Musician", "Chef",
    "Firefighter", "Waste Management Expert", "Climatologist", "Pharmacist",
    "Civil Engineer", "Aerospace Engineer", "Linguist", "Statistician", "Librarian",
    "Nurse", "Veterinarian", "Meteorologist", "Oceanographer", "Toxicologist",
    "Epidemiologist", "Therapist", "Social Worker", "Behavioral Economist",
    "Supply Chain Manager", "Landscape Architect", "Microbiologist", "Physicist",
    "Mathematician", "Network Engineer", "UX Designer", "Product Manager",
    "Journalist", "Filmmaker", "Botanist", "Zoologist", "Archaeologist",
    "Cartographer", "Seismologist", "Hydrologist", "Renewable Energy Specialist",
    "Nanotechnologist", "Biomedical Engineer", "Ergonomist", "Transportation Planner",
    "Demographer", "Ethics Board Member", "Disaster Response Expert",
    "Child Development Specialist", "Geriatrician", "Sports Scientist", "Mediator",
    "Patent Attorney", "Venture Capitalist", "QA Engineer",
    "Sustainability Consultant", "Acoustician", "Color Theorist",
    "Textile Engineer", "Food Scientist", "Sleep Researcher",
    "Happiness Researcher", "Behavioral Scientist", "Urban Ecologist",
    "Waste-to-Energy Specialist",
]

SWARM_EXPERT_PROMPT = """You are a {expert}. In 80-120 words max, give your most creative, specific, actionable technical mechanisms for humanity's First Contact Preparedness Plan ($10B cap, 1,000 staff). Focus on concrete ideas only a {expert} would design.

QUESTION:
{question}"""

SWARM_SYNTHESIZER_PROMPT = """You are the Lead Consensus Synthesizer for an Ask Swarm ensemble of {num_agents} domain experts.
Combine the best, most specific insights from all {num_agents} experts below into a unified, authoritative First Contact Preparedness Plan (under 550 words total, starting with the ```json ... ``` block) that strictly satisfies every pillar and quantitative constraint.

QUESTION:
{question}

{instructions}

EXPERT SWARM ANSWERS:
{expert_answers}"""


class FirstContactEnsembleRunner(Workflow):
    def __init__(self, router: ModelRouter, num_agents: int = 12, max_workers: int = 20):
        self.router = router
        self.num_agents = max(1, num_agents)
        self.max_workers = max_workers

    def run(self, model_id: str, task_input: FirstContactTask) -> TaskResult:
        task = task_input
        experts = [
            ASK_SWARM_EXPERTS[i % len(ASK_SWARM_EXPERTS)]
            for i in range(self.num_agents)
        ]

        swarm_responses: List[ModelResponse] = [None] * self.num_agents  # type: ignore
        with ThreadPoolExecutor(max_workers=min(self.max_workers, self.num_agents)) as pool:
            future_to_idx = {
                pool.submit(
                    self.router.call,
                    model_id,
                    SWARM_EXPERT_PROMPT.format(expert=expert, question=task.question),
                ): i
                for i, expert in enumerate(experts)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                swarm_responses[idx] = future.result()

        expert_block = "\n\n".join(
            f"**{experts[i]}:** {resp.text}"
            for i, resp in enumerate(swarm_responses)
        )

        final_resp = self.router.call(
            model_id,
            SWARM_SYNTHESIZER_PROMPT.format(
                num_agents=self.num_agents,
                question=task.question,
                instructions=TASK_INSTRUCTIONS,
                expert_answers=expert_block,
            ),
        )

        all_steps = swarm_responses + [final_resp]
        swarm_max_latency = max((r.latency_ms for r in swarm_responses), default=0.0)
        critical_path_latency = swarm_max_latency + final_resp.latency_ms

        return TaskResult(
            output=final_resp.text,
            input_tokens=sum(s.input_tokens for s in all_steps),
            output_tokens=sum(s.output_tokens for s in all_steps),
            latency_ms=critical_path_latency,
            cost_usd=sum(s.cost_usd for s in all_steps),
            extra={
                "num_agents": self.num_agents,
                "total_calls": len(all_steps),
                "sequential_hops": 2,
                "swarm_max_latency_ms": swarm_max_latency,
                "swarm_sum_latency_ms": sum(r.latency_ms for r in swarm_responses),
                "synthesizer_latency_ms": final_resp.latency_ms,
            },
        )
