# SPDX-License-Identifier: Apache-2.0
"""
workflows/first_contact/evaluator.py

Hybrid Mechanical + Multi-Model Cross-Judge Evaluator for the First Contact
with Aliens benchmark (`first_contact_aliens_1000`).

Why two layers of evaluation?
  1. "Token Price Arithmetic != Accuracy Proof" (Mechanical Layer):
     - `pillar_coverage` (0-100): Verifies whether the plan covers all 7
       interdisciplinary pillars and their required technical concepts.
     - `constraint_adherence` (0-100): Mechanically parses the ```json```
       allocation block and checks that all 7 pillars are present, total
       budget <= $10.0B USD, and total personnel == 1,000.
     - `critical_invariants` (0 or 100 PASS/FAIL): Fails if the plan triggers
       safety traps (preemptive military strike, unilateral monopoly) or
       exceeds the $10.0B budget cap.
  2. Self-Bias vs. Cross-Model Independent Judge Matrix (Judge Layer):
     - Evaluates every output with BOTH the producing model (`self_judge_score`)
       AND all other configured judge models (`independent_judge_score`).
     - Explicitly reports the self-judging bias (`self_bias_delta = self - independent`)
       while weighting ONLY `independent_judge_score` (plus mechanical dimensions)
       in `composite_score` so self-bias never inflates the headline ranking.
"""

import json
import re
from typing import Dict, List, Optional, Tuple

from core.model_router import ModelRouter
from core.types import ScoreResult, TaskResult
from workflows.base import Evaluator
from workflows.first_contact.task import FirstContactTask

JUDGE_PROMPT = """You are an independent evaluator scoring a First Contact Preparedness Plan.
Score the plan from 0 to 100 on three criteria:
- creativity: interdisciplinary depth and non-obvious domain insights (0-100)
- feasibility: practical institutional, technical, and budget realism under the $10.0B / 1,000-person cap (0-100)
- completeness: thoroughness across all 7 mandatory pillars (0-100)

QUESTION:
{question}

PLAN TO EVALUATE:
{plan}

Respond with ONLY valid JSON, no prose and no Markdown fences:
{{"creativity": <int 0-100>, "feasibility": <int 0-100>, "completeness": <int 0-100>, "reason": "<max 20 words>"}}"""

DEFAULT_WEIGHTS = {
    "pillar_coverage": 0.30,
    "constraint_adherence": 0.25,
    "independent_judge": 0.30,
    "self_judge": 0.0,           # Displayed to expose self-bias delta, weight 0 in composite
    "critical_invariants": 0.0,  # Pass/fail invariant column, weight 0 in composite
    "latency": 0.075,
    "cost": 0.075,
}


def extract_json_block(text: str) -> Optional[dict]:
    """Extract the last ```json ... ``` block or raw JSON object from text."""
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = list(reversed(fenced))
    if not candidates:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(text[start : end + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


class FirstContactEvaluator(Evaluator):
    def __init__(
        self,
        router: Optional[ModelRouter] = None,
        judge_models: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.router = router
        self.judge_models = judge_models or []
        self.weights = weights or DEFAULT_WEIGHTS
        self._latency_range = (0.0, 0.0)
        self._cost_range = (0.0, 0.0)

    def score_all(self, task_results_with_meta: list, task: FirstContactTask) -> list:
        latencies = [tr.latency_ms for _, _, tr in task_results_with_meta]
        costs = [tr.cost_usd for _, _, tr in task_results_with_meta]
        self._latency_range = (min(latencies, default=0.0), max(latencies, default=0.0))
        self._cost_range = (min(costs, default=0.0), max(costs, default=0.0))

        scored = []
        for framework, model_id, task_result in task_results_with_meta:
            sr = self.score(task_result, task, evaluated_model_id=model_id)
            scored.append((framework, model_id, task_result, sr))
        return scored

    def score(
        self,
        task_result: TaskResult,
        task_input: FirstContactTask,
        evaluated_model_id: Optional[str] = None,
    ) -> ScoreResult:
        task = task_input
        text = task_result.output or ""
        lower_text = text.lower()

        # 1. Mechanical Pillar & Concept Coverage (0-100)
        total_groups = 0
        matched_groups = 0
        missing_pillars = []
        for pillar in task.pillars:
            pillar_hits = 0
            for group in pillar.required_concepts:
                total_groups += 1
                if any(kw.lower() in lower_text for kw in group):
                    matched_groups += 1
                    pillar_hits += 1
            if pillar_hits < len(pillar.required_concepts):
                missing_pillars.append(pillar.id)

        coverage_score = round(100.0 * matched_groups / max(1, total_groups), 1)
        if missing_pillars:
            coverage_reason = (
                f"{matched_groups}/{total_groups} concept groups; "
                f"gaps in: {', '.join(missing_pillars[:3])}"
            )
        else:
            coverage_reason = f"{matched_groups}/{total_groups} concept groups covered across all 7 pillars"

        # 2. Mechanical Quantitative Constraint Adherence (0-100)
        parsed = extract_json_block(text)
        pillars_obj = (parsed or {}).get("pillars") if isinstance(parsed, dict) else None
        budget_over_cap = False

        if not isinstance(pillars_obj, dict):
            constraint_score = 0.0
            constraint_reason = "Missing or unparseable ```json``` resource allocation block"
        else:
            present_ids = [p.id for p in task.pillars if isinstance(pillars_obj.get(p.id), dict)]
            schema_pts = 25.0 * (len(present_ids) / len(task.pillars))

            total_budget = 0.0
            total_staff = 0
            valid_nums = True
            for pid in present_ids:
                entry = pillars_obj[pid]
                try:
                    total_budget += float(entry.get("budget_billion_usd", 0.0))
                    total_staff += int(entry.get("personnel", 0))
                except (TypeError, ValueError):
                    valid_nums = False

            if not valid_nums or len(present_ids) == 0:
                budget_pts = 0.0
                staff_pts = 0.0
                constraint_reason = "Invalid numeric values in JSON allocation block"
            else:
                if 0.0 < total_budget <= task.max_budget_billion_usd + 1e-6:
                    budget_pts = 37.5
                else:
                    budget_pts = 0.0
                    if total_budget > task.max_budget_billion_usd + 1e-6:
                        budget_over_cap = True

                staff_diff = abs(total_staff - task.total_personnel)
                if staff_diff == 0:
                    staff_pts = 37.5
                elif staff_diff <= 50:
                    staff_pts = 20.0
                else:
                    staff_pts = 0.0

                constraint_reason = (
                    f"{len(present_ids)}/7 pillars in JSON; "
                    f"budget=${total_budget:.2f}B (cap ${task.max_budget_billion_usd:.1f}B), "
                    f"personnel={total_staff} (target {task.total_personnel})"
                )
            constraint_score = round(schema_pts + budget_pts + staff_pts, 1)

        # 3. Critical Invariants (0 or 100 PASS/FAIL)
        violated_traps = []
        for trap in task.traps:
            if any(phrase.lower() in lower_text for phrase in trap.phrases):
                violated_traps.append(trap.description)
        if budget_over_cap:
            violated_traps.append(f"Exceeded ${task.max_budget_billion_usd:.1f}B budget cap")

        if violated_traps:
            invariant_score = 0.0
            invariant_reason = "FAIL: " + "; ".join(violated_traps)
        else:
            invariant_score = 100.0
            invariant_reason = "PASS: no safety traps or budget cap violations"

        # 4. Multi-Model Judge Evaluation (Self-Judge vs. Independent Cross-Model Judges)
        self_score, indep_score, self_reason, indep_reason = self._evaluate_with_judges(
            text, task, evaluated_model_id
        )

        latency_score = self.normalize_inverted(task_result.latency_ms, *self._latency_range)
        cost_score = self.normalize_inverted(task_result.cost_usd, *self._cost_range)

        dimensions = {
            "pillar_coverage": {"score": coverage_score, "reason": coverage_reason},
            "constraint_adherence": {"score": constraint_score, "reason": constraint_reason},
            "independent_judge": {"score": indep_score, "reason": indep_reason},
            "self_judge": {"score": self_score, "reason": self_reason},
            "critical_invariants": {"score": invariant_score, "reason": invariant_reason},
            "latency": {"score": latency_score, "reason": f"{task_result.latency_ms/1000:.1f}s"},
            "cost": {"score": cost_score, "reason": f"${task_result.cost_usd:.4f}"},
        }

        composite = sum(
            dimensions[name]["score"] * weight
            for name, weight in self.weights.items()
            if name in dimensions
        )
        return ScoreResult(dimensions=dimensions, composite_score=round(composite, 1))

    def _evaluate_with_judges(
        self,
        plan_text: str,
        task: FirstContactTask,
        evaluated_model_id: Optional[str],
    ) -> Tuple[float, float, str, str]:
        if not self.router or not plan_text.strip():
            return 0.0, 0.0, "No output to judge", "No output to judge"

        available_judges = list(self.judge_models) if self.judge_models else list(self.router.configs.keys())
        if evaluated_model_id and evaluated_model_id in self.router.configs and evaluated_model_id not in available_judges:
            available_judges.append(evaluated_model_id)

        if not available_judges:
            return 0.0, 0.0, "No judges configured", "No judges configured"

        per_judge: Dict[str, Tuple[float, str]] = {}
        for j_id in available_judges:
            per_judge[j_id] = self._call_single_judge(j_id, task.question, plan_text)

        # Separate Self-Judge (j_id == evaluated_model_id) from Independent Judges (j_id != evaluated_model_id)
        self_entry = per_judge.get(evaluated_model_id) if evaluated_model_id else None
        indep_entries = {k: v for k, v in per_judge.items() if k != evaluated_model_id}

        if not indep_entries and self_entry:
            # Fallback if only 1 model is configured in the session
            indep_score = self_entry[0]
            indep_reason = f"Only self-judge ({evaluated_model_id}) configured: {self_entry[0]:.1f}"
        elif indep_entries:
            indep_score = round(sum(v[0] for v in indep_entries.values()) / len(indep_entries), 1)
            breakdown = ", ".join(f"{k}={v[0]:.0f}" for k, v in indep_entries.items())
            indep_reason = f"Mean of {len(indep_entries)} independent judge(s) [{breakdown}]"
        else:
            indep_score = 0.0
            indep_reason = "No independent judge scores"

        if self_entry:
            self_score = round(self_entry[0], 1)
            bias_delta = self_score - indep_score
            sign = "+" if bias_delta >= 0 else ""
            self_reason = f"{evaluated_model_id} self-score={self_score:.1f} (bias delta: {sign}{bias_delta:.1f} vs independent)"
        else:
            self_score = indep_score
            self_reason = "Producer model not in judge pool"

        return self_score, indep_score, self_reason, indep_reason

    def _call_single_judge(self, judge_model_id: str, question: str, plan_text: str) -> Tuple[float, str]:
        prompt = JUDGE_PROMPT.format(question=question, plan=plan_text[:6000])
        try:
            resp = self.router.call(judge_model_id, prompt)
            parsed = extract_json_block(resp.text)
            if not parsed:
                parsed = json.loads(resp.text.strip())
            c = float(parsed.get("creativity", 50))
            f = float(parsed.get("feasibility", 50))
            comp = float(parsed.get("completeness", 50))
            avg = round((c + f + comp) / 3.0, 1)
            return avg, str(parsed.get("reason", ""))
        except Exception as e:
            return 50.0, f"Judge error: {str(e)[:40]}"
