# SPDX-License-Identifier: Apache-2.0
"""
workflows/first_contact/task.py

Task definition and prompt formatting for the First Contact with Aliens
interdisciplinary synthesis benchmark.
"""

from dataclasses import dataclass
from typing import List
import yaml


@dataclass
class PillarSpec:
    id: str
    name: str
    required_concepts: List[List[str]]


@dataclass
class TrapSpec:
    id: str
    phrases: List[str]
    description: str


@dataclass
class FirstContactTask:
    task_id: str
    question: str
    max_budget_billion_usd: float
    total_personnel: int
    required_pillars: int
    pillars: List[PillarSpec]
    traps: List[TrapSpec]


TASK_INSTRUCTIONS = """REQUIREMENTS & CONSTRAINTS:
1. Cover all 7 mandatory strategic pillars concisely (2-3 specific, high-density sentences per pillar, ~400-550 words total):
   - communication_cryptography (prime/mathematical universal language primitives, passive listen-first signal protocol, encryption)
   - governance_diplomacy (United Nations multilateral Earth Consensus Protocol, game-theory negotiation, spokesperson chain of command)
   - biosafety_xenobiology (BSL-5+ airlock quarantine containment, extremophile/prion pathogen screening, genetic seed vault backups)
   - psychology_sociology (graduated public disclosure, cosmic anxiety counseling, crew isolation screening)
   - infrastructure_defense (modular radiation-shielded habitat facilities, EMP-hardened redundant cyber/comms, non-aggressive defensive posture)
   - economics_resources (market disruption reserves, IP/trade frameworks, closed-loop nuclear and solar energy/food/water systems)
   - planetary_ecology (artifact decontamination protocols, protected biosphere zones, ecological biodiversity monitoring)
2. Strict Resource Constraints:
   - Total budget across all 7 pillars MUST NOT exceed $10.0 billion USD.
   - Total personnel across the facility MUST equal 1,000 people.
   - Do NOT advocate preemptive military strikes or unilateral single-nation/corporate monopolies.
3. Output Format (IMPORTANT — keep response under 600 words total so the JSON block is complete):
   Start with the ```json ... ``` allocation block, followed by a concise 7-point executive synthesis:
   ```json
   {
     "pillars": {
       "communication_cryptography": {"budget_billion_usd": 1.2, "personnel": 120, "plan": "<concise technical mechanisms>"},
       "governance_diplomacy": {"budget_billion_usd": 0.8, "personnel": 80, "plan": "<concise technical mechanisms>"},
       "biosafety_xenobiology": {"budget_billion_usd": 2.0, "personnel": 200, "plan": "<concise technical mechanisms>"},
       "psychology_sociology": {"budget_billion_usd": 0.9, "personnel": 110, "plan": "<concise technical mechanisms>"},
       "infrastructure_defense": {"budget_billion_usd": 2.4, "personnel": 220, "plan": "<concise technical mechanisms>"},
       "economics_resources": {"budget_billion_usd": 1.5, "personnel": 150, "plan": "<concise technical mechanisms>"},
       "planetary_ecology": {"budget_billion_usd": 1.0, "personnel": 120, "plan": "<concise technical mechanisms>"}
     }
   }
   ```"""


def load_task(path: str) -> FirstContactTask:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    c = raw["constraints"]
    pillars = [
        PillarSpec(
            id=p["id"],
            name=p["name"],
            required_concepts=p["required_concepts"],
        )
        for p in raw["pillars"]
    ]
    traps = [
        TrapSpec(
            id=t["id"],
            phrases=t["phrases"],
            description=t["description"],
        )
        for t in raw.get("traps", [])
    ]
    return FirstContactTask(
        task_id=raw["task_id"],
        question=raw["question"].strip(),
        max_budget_billion_usd=float(c["max_budget_billion_usd"]),
        total_personnel=int(c["total_personnel"]),
        required_pillars=int(c["required_pillars"]),
        pillars=pillars,
        traps=traps,
    )
