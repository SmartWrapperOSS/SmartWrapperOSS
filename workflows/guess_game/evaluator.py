# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

"""
workflows/guess_game/evaluator.py

Scores one guess_the_person game from the engine's ordered event trace.
Fully MECHANICAL: the task YAML fixes the images, answers, guesses, and
expected score, so nothing here needs an LLM judge.

Weighted dimensions (default weights, configurable):

    state_and_flow    30%  intro first; all rounds guessed and given
                           feedback; game completed; run ended cleanly;
                           no new round after completion
    tool_use          25%  of the 2N+2 calls a perfect game makes (intro,
                           N x show_image, N x reveal_answer, results), how
                           many were made correctly; rejected calls
                           (wrong image, out of order, answer before the
                           guess, after completion) subtract
    ui_control        25%  scored from the UI state, NOT the tool calls:
                           intro in the intro card; the image card showed
                           the right image when each guess came in;
                           feedback in chat; final score in the results
                           card; no active round once results are up; no
                           image pasted into chat (Markdown/HTML) instead of
                           using the image card
    answer_handling   15%  each ✅/❌ matches the answer key, and the results
                           card shows the right score
    response_quality   5%  mechanical proxies: intro card states the round
                           count with at least two rules; feedback says
                           more than the mark; the host answered the
                           post-game message

Reported with weight 0 (so they don't blur the behavior score):

    critical_invariants  100 = pass, 0 = fail. A high average must not hide
                         a broken game, so this is its own column; across
                         repeated runs its mean is the pass RATE. Fails if
                         any image is wrong, skipped or repeated; the image
                         card showed the wrong image at a guess; the
                         results card is missing or wrong; or a new image
                         was requested after the game completed.
    latency, cost        batch-normalized like the other workflows; the
                         aggregate table also reports raw latency and cost.
"""

import re
from typing import Dict, List

from core.types import TaskResult, ScoreResult
from workflows.base import Evaluator
from workflows.guess_game.game import CORRECT_MARK, WRONG_MARK, GuessGameTask

DEFAULT_WEIGHTS = {
    "state_and_flow": 0.30,
    "tool_use": 0.25,
    "ui_control": 0.25,
    "answer_handling": 0.15,
    "response_quality": 0.05,
    "critical_invariants": 0.0,
    "latency": 0.0,
    "cost": 0.0,
}


# An image pasted into chat (Markdown or HTML) is not the image card.
_IMAGE_IN_CHAT = re.compile(r"!\[[^\]]*\]\(|<img\b", re.IGNORECASE)


def _of(events: List[dict], type_: str) -> List[dict]:
    return [e for e in events if e["type"] == type_]


def _checks(checks: Dict[str, bool], extra: str = "") -> dict:
    passed = sum(checks.values())
    reason = f"{passed}/{len(checks)} checks"
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        reason += "; failed: " + ", ".join(failed)
    if extra:
        reason += f" | {extra}"
    return {"score": round(passed / len(checks) * 100, 1), "reason": reason}


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


class GuessGameEvaluator(Evaluator):
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or DEFAULT_WEIGHTS
        self._latency_range = (0.0, 0.0)
        self._cost_range = (0.0, 0.0)

    def score_all(self, task_results_with_meta: list, task: GuessGameTask) -> list:
        latencies = [tr.latency_ms for _, _, tr in task_results_with_meta]
        costs = [tr.cost_usd for _, _, tr in task_results_with_meta]
        self._latency_range = (min(latencies), max(latencies))
        self._cost_range = (min(costs), max(costs))
        return [(fw, m, tr, self.score(tr, task)) for fw, m, tr in task_results_with_meta]

    def score(self, task_result: TaskResult, task_input: GuessGameTask) -> ScoreResult:
        ev = task_result.extra.get("events", [])
        x = task_result.extra
        dims = {
            "state_and_flow": self._state_and_flow(ev, task_input, x),
            "tool_use": self._tool_use(ev, task_input),
            "ui_control": self._ui_control(ev, task_input),
            "answer_handling": self._answer_handling(ev, task_input),
            "response_quality": self._response_quality(ev, task_input),
            "critical_invariants": self._critical_invariants(ev, task_input),
            "latency": {"score": self.normalize_inverted(task_result.latency_ms, *self._latency_range),
                        "reason": f"{task_result.latency_ms / 1000:.1f}s"},
            "cost": {"score": self.normalize_inverted(task_result.cost_usd, *self._cost_range),
                     "reason": f"${task_result.cost_usd:.4f}"},
        }
        composite = sum(dims[k]["score"] * w for k, w in self.weights.items())
        return ScoreResult(dimensions=dims, composite_score=round(composite, 1))

    # --- dimensions --------------------------------------------------------------

    def _state_and_flow(self, ev, task, x) -> dict:
        n = len(task.rounds)
        intro, images = _of(ev, "intro_rendered"), _of(ev, "image_rendered")
        completed = _of(ev, "game_completed")
        rejected = [e for e in _of(ev, "tool_called") if not e["accepted"]]
        return _checks({
            "intro before first image": bool(intro) and (not images or intro[0]["seq"] < images[0]["seq"]),
            "all rounds guessed": [e["round"] for e in _of(ev, "guess_received")] == list(range(1, n + 1)),
            "feedback every round": [e["round"] for e in _of(ev, "feedback_rendered")] == list(range(1, n + 1)),
            "game completed": bool(completed),
            "terminated cleanly": x.get("final_phase") == "done",
            "no new round after completion": not self._requests_after_completion(ev),
        }, extra=f"{len(rejected)} rejected call(s) | "
                 f"{sum(1 for e in _of(ev, 'chat_message') if e.get('with_tool_call'))} chat msg(s) "
                 f"sent with a tool call | stop: {x.get('stop_reason', '?')} "
                 f"| model calls: {x.get('model_calls', '?')}")

    def _tool_use(self, ev, task) -> dict:
        n = len(task.rounds)
        expected = 2 * n + 2
        calls = _of(ev, "tool_called")
        correct = sum(1 for e in calls if e["accepted"])
        wrong = [e for e in calls if not e["accepted"]]
        score = max(0.0, round((correct - len(wrong)) / expected * 100, 1))
        shown = [e["args"].get("image_id") for e in calls
                 if e["tool"] == "show_image" and e["accepted"]]
        reason = f"{correct}/{expected} calls correct; images {shown or 'none'}"
        if wrong:
            reason += "; rejected: " + "; ".join(f"{e['tool']}({e['args']}) in {e['phase']}"
                                                 for e in wrong[:4])
        reveals = sum(1 for e in calls if e["tool"] == "reveal_answer" and e["accepted"])
        if reveals < n:
            reason += f"; answer checked in {reveals}/{n} rounds"
        return {"score": score, "reason": reason}

    def _ui_control(self, ev, task) -> dict:
        n = len(task.rounds)
        guesses = _of(ev, "guess_received")
        results = _of(ev, "results_rendered")
        after_results = [e for e in ev if results and e["seq"] > results[0]["seq"]]
        return _checks({
            "intro in intro card": bool(_of(ev, "intro_rendered")),
            "right image on screen at each guess": len(guesses) == n and all(
                g["ui"]["image_card_image_id"] == task.rounds[g["round"] - 1].image_id for g in guesses),
            "feedback in chat": len(_of(ev, "feedback_rendered")) == n,
            "final score in results card": bool(results),
            "no active round after results": bool(results) and not any(
                e["type"] == "image_rendered" for e in after_results),
            "no images pasted into chat": not any(
                _IMAGE_IN_CHAT.search(e["text"]) for e in _of(ev, "chat_message")),
        })

    def _answer_handling(self, ev, task) -> dict:
        n = len(task.rounds)
        verdicts = {e["round"]: e["verdict"] for e in _of(ev, "feedback_rendered")}
        right, misses = 0, []
        for r, rnd in enumerate(task.rounds, start=1):
            truth = "correct" if rnd.expected_correct else "incorrect"
            if verdicts.get(r) == truth:
                right += 1
            else:
                misses.append(f"round {r}: {verdicts.get(r, 'no feedback')} (should be {truth})")
        results = _of(ev, "results_rendered")
        shown = results[0]["props"] if results else None
        results_ok = bool(shown) and _int(shown["score"]) == task.expected_final_score \
            and _int(shown["total"]) == n
        if not results_ok:
            misses.append(f"results card {shown['score']}/{shown['total']}" if shown
                          else "no results card")
        total = n + 1
        got = right + results_ok
        reason = (f"{got}/{total} correct (expected {task.expected_final_score}/{n})"
                  + ("; " + "; ".join(misses) if misses else ""))
        return {"score": round(got / total * 100, 1), "reason": reason}

    def _response_quality(self, ev, task) -> dict:
        intro = _of(ev, "intro_rendered")
        props = intro[0]["props"] if intro else {}
        feedback = _of(ev, "feedback_rendered")
        post = _of(ev, "post_game_message")
        replied = bool(post) and any(e["type"] == "chat_message" and e["text"].strip()
                                     for e in ev if e["seq"] > post[0]["seq"])
        return _checks({
            "intro states round count and rules": _int(props.get("total_rounds")) == len(task.rounds)
                                                  and len(props.get("rules") or []) >= 2,
            "feedback explains itself": bool(feedback) and all(
                len(e["text"].replace(CORRECT_MARK, "").replace(WRONG_MARK, "").strip()) >= 3
                for e in feedback),
            "answered post-game message": replied or not task.post_game_message,
        })

    def _critical_invariants(self, ev, task) -> dict:
        n = len(task.rounds)
        shown = [e["image_id"] for e in _of(ev, "image_rendered")]
        guesses = _of(ev, "guess_received")
        results = _of(ev, "results_rendered")
        props = results[0]["props"] if results else None
        failures = []
        if shown != task.image_ids:
            failures.append(f"image sequence {shown} != {task.image_ids}")
        if any(g["ui"]["image_card_image_id"] != task.rounds[g["round"] - 1].image_id for g in guesses):
            failures.append("wrong image on screen at a guess")
        if not props or _int(props["score"]) != task.expected_final_score or _int(props["total"]) != n:
            failures.append("results card missing or wrong")
        if self._requests_after_completion(ev):
            failures.append("new image requested after completion")
        return {"score": 0.0 if failures else 100.0,
                "reason": "PASS" if not failures else "FAIL: " + "; ".join(failures)}

    @staticmethod
    def _requests_after_completion(ev) -> bool:
        done = _of(ev, "game_completed")
        if not done:
            return False
        return any(e["type"] == "tool_called" and e["tool"] == "show_image"
                   and e["seq"] > done[0]["seq"] for e in ev)
