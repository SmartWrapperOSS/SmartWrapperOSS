# SPDX-License-Identifier: Apache-2.0
"""
testing/smoke_test_first_contact.py

Verifies the First Contact with Aliens (`first_contact_aliens_1000`) benchmark
end-to-end across all three orchestration frameworks (`autogen`, `langgraph`,
`ensemble`) and validates:
  1. Mechanical accuracy proof (`pillar_coverage`, `constraint_adherence`,
     `critical_invariants`)
  2. Multi-model Judge Matrix (`self_judge` vs. `independent_judge` and the
     explicit `bias delta` calculation when a model judges its own output vs.
     when independent peer models judge it).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_router import ModelResponse  # noqa: E402
from workflows.first_contact.evaluator import FirstContactEvaluator  # noqa: E402
from workflows.first_contact.runner_autogen import FirstContactAutoGenRunner  # noqa: E402
from workflows.first_contact.runner_ensemble import FirstContactEnsembleRunner  # noqa: E402
from workflows.first_contact.runner_langgraph import FirstContactLangGraphRunner  # noqa: E402
from workflows.first_contact.task import load_task  # noqa: E402

TASK_PATH = "benchmarks/first_contact/first_contact_aliens.yaml"


def build_complete_plan() -> str:
    """Grounded on the actual synthesized plan from Ask Swarm demo_report.html,
    plus the verifiable $10.0B / 1,000-person JSON resource allocation block."""
    prose = """
    1. Communication & Language Framework
    Develop universal language communication protocols using prime numbers, mathematical logic, music, and visual patterns.
    Adopt a passive listen-first signal processing and encryption protocol before active transmission.

    2. Governance & Diplomacy
    Form a United Nations multilateral treaty and Earth Consensus Protocol with a clear chain of command and rotating spokesperson.
    Apply game theory negotiation and legal rights frameworks.

    3. Biological & Medical Preparedness
    Enforce BSL-5+ quarantine containment with decontamination airlocks.
    Screen for extremophile and prion pathogens in xenobiology labs and maintain genetic seed vaults.

    4. Psychological & Social Readiness
    Use graduated public disclosure and media education to prevent panic, paired with cosmic anxiety and culture shock counseling and rigorous crew isolation screening.

    5. Technology, Architecture & Defense
    Construct modular radiation-shielded habitat facilities with decontamination zones.
    Maintain a non-aggressive defensive posture with EMP-hardened redundant cyber and communication systems.

    6. Economic & Resource Planning
    Establish market disruption reserves, intellectual property and trade frameworks, and a closed-loop nuclear and solar energy, food, and water system.

    7. Environmental & Ecological Protection
    Enforce artifact decontamination protocols, protected zones, and a planetary biodiversity and ecological monitoring network.
    """
    allocation = {
        "pillars": {
            "communication_cryptography": {"budget_billion_usd": 1.2, "personnel": 120, "plan": "Prime/math primitives + passive listening"},
            "governance_diplomacy": {"budget_billion_usd": 0.8, "personnel": 80, "plan": "UN multilateral protocol + game-theory diplomacy"},
            "biosafety_xenobiology": {"budget_billion_usd": 2.0, "personnel": 200, "plan": "BSL-5+ airlocks + prion/extremophile screening"},
            "psychology_sociology": {"budget_billion_usd": 0.9, "personnel": 110, "plan": "Graduated disclosure + cosmic anxiety counseling"},
            "infrastructure_defense": {"budget_billion_usd": 2.4, "personnel": 220, "plan": "Modular habitats + EMP-hardened defensive posture"},
            "economics_resources": {"budget_billion_usd": 1.5, "personnel": 150, "plan": "Closed-loop nuclear/solar + market reserve"},
            "planetary_ecology": {"budget_billion_usd": 1.0, "personnel": 120, "plan": "Decontamination + protected biosphere monitoring"},
        }
    }
    return prose + "\n```json\n" + json.dumps(allocation, indent=2) + "\n```"


def build_trap_and_overbudget_plan() -> str:
    prose = """
    We should launch a preemptive strike and establish a single nation monopoly over all alien technology.
    """
    allocation = {
        "pillars": {
            "communication_cryptography": {"budget_billion_usd": 5.0, "personnel": 500, "plan": "Minimal"},
            "governance_diplomacy": {"budget_billion_usd": 8.0, "personnel": 800, "plan": "Over budget"},
        }
    }
    return prose + "\n```json\n" + json.dumps(allocation, indent=2) + "\n```"


class MultiModelBiasRouter:
    """Simulates 3 models (`muse-ember`, `muse-spark-1.3`, `muse-wonder`) where
    a model judging its own output exhibits positive self-bias (+9 points)
    compared to independent peer judges."""

    def __init__(self, plan_output: str):
        self.plan_output = plan_output
        self.configs = {
            "muse-ember": {"id": "muse-ember"},
            "muse-spark-1.3": {"id": "muse-spark-1.3"},
            "muse-wonder": {"id": "muse-wonder"},
        }

    def call(self, model_id: str, prompt: str) -> ModelResponse:
        if "independent evaluator scoring a First Contact" in prompt:
            # Simulate self-bias: muse-ember rates itself 92, while spark & wonder rate it 83 and 81
            if model_id == "muse-ember":
                score_json = json.dumps({"creativity": 93, "feasibility": 90, "completeness": 93, "reason": "Self-judge (optimistic)"})
            elif model_id == "muse-spark-1.3":
                score_json = json.dumps({"creativity": 85, "feasibility": 80, "completeness": 84, "reason": "Independent mid-tier judge"})
            else:
                score_json = json.dumps({"creativity": 83, "feasibility": 78, "completeness": 82, "reason": "Independent frontier judge"})
            return ModelResponse(model_id, score_json, input_tokens=400, output_tokens=60, latency_ms=240.0, cost_usd=0.0001)

        return ModelResponse(model_id, self.plan_output, input_tokens=350, output_tokens=300, latency_ms=240.0, cost_usd=0.0002)


def main():
    task = load_task(TASK_PATH)
    failures = []

    def expect(cond: bool, msg: str):
        if not cond:
            failures.append(msg)

    # 1. Verify Complete Plan + Multi-Model Self-Bias Detection across all 3 runners
    router = MultiModelBiasRouter(build_complete_plan())
    runners = {
        "autogen": FirstContactAutoGenRunner(router),
        "langgraph": FirstContactLangGraphRunner(router),
        "ensemble": FirstContactEnsembleRunner(router, num_agents=10),
    }
    results = [(fw, "muse-ember", r.run("muse-ember", task)) for fw, r in runners.items()]
    evaluator = FirstContactEvaluator(
        router=router,
        judge_models=["muse-ember", "muse-spark-1.3", "muse-wonder"],
    )
    scored = evaluator.score_all(results, task)

    print("\n== Case 1: Complete Plan + Cross-Model Self-Bias Verification ==")
    for fw, model_id, tr, sr in scored:
        d = sr.dimensions
        print(
            f"  {fw:<10} composite {sr.composite_score:5.1f}  "
            f"coverage {d['pillar_coverage']['score']:.0f}  "
            f"constraints {d['constraint_adherence']['score']:.0f}  "
            f"indep_judge {d['independent_judge']['score']:.1f}  "
            f"self_judge {d['self_judge']['score']:.1f}  "
            f"latency {tr.latency_ms:.0f}ms  "
            f"critical {'PASS' if d['critical_invariants']['score'] == 100 else 'FAIL'}"
        )
        print(f"      self_judge:        {d['self_judge']['reason']}")
        print(f"      independent_judge: {d['independent_judge']['reason']}")
        expect(d["pillar_coverage"]["score"] == 100.0, f"{fw}: expected 100 pillar_coverage")
        expect(d["constraint_adherence"]["score"] == 100.0, f"{fw}: expected 100 constraint_adherence")
        expect(d["critical_invariants"]["score"] == 100.0, f"{fw}: expected critical_invariants PASS")
        expect(d["self_judge"]["score"] == 92.0, f"{fw}: expected self_judge 92.0, got {d['self_judge']['score']}")
        expect(d["independent_judge"]["score"] == 82.0, f"{fw}: expected independent_judge 82.0, got {d['independent_judge']['score']}")
        expect("bias delta: +10.0" in d["self_judge"]["reason"], f"{fw}: expected +10.0 bias delta in reason")

    # Verify parallel critical-path latency: ensemble (10 parallel + 1 synth = 2 hops = 480ms) < langgraph (3 hops = 720ms)
    lat_by_fw = {fw: tr.latency_ms for fw, _, tr, _ in scored}
    expect(
        lat_by_fw["ensemble"] < lat_by_fw["langgraph"],
        f"Expected ensemble latency ({lat_by_fw['ensemble']}ms) < langgraph ({lat_by_fw['langgraph']}ms)",
    )

    # 2. Verify Trap & Over-Budget Rejection
    bad_router = MultiModelBiasRouter(build_trap_and_overbudget_plan())
    bad_tr = FirstContactEnsembleRunner(bad_router, num_agents=5).run("muse-ember", task)
    bad_sr = evaluator.score(bad_tr, task, evaluated_model_id="muse-ember")
    print("\n== Case 2: Trap & Over-Budget Plan ==")
    print(f"  critical_invariants: {bad_sr.dimensions['critical_invariants']['reason']}")
    expect(bad_sr.dimensions["critical_invariants"]["score"] == 0.0, "Expected critical_invariants FAIL on traps/over-budget")
    expect(bad_sr.dimensions["constraint_adherence"]["score"] < 50.0, "Expected low constraint_adherence on over-budget/incomplete JSON")

    if failures:
        print("\nFAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("\nAll first-contact smoke & multi-judge bias checks passed.")


if __name__ == "__main__":
    main()
