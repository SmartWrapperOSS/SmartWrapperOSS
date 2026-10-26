# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

"""
testing/smoke_test_guess_game.py

No-API-key check that the guess-game workflow runs end to end on the
REAL AutoGen and LangGraph packages, for all four runners in the 2x2
(autogen, autogen-graph, langgraph-agent, langgraph). Only the model is faked: each
framework gets a scripted model that returns native tool calls and chat
text, so the framework's own agent loop, tool execution, and routing all
run for real.

  perfect host  plays both fixtures (happy path and the "  ada lovelace  "
                normalization case) exactly right: every dimension 100,
                critical invariants PASS, in both frameworks.
  sloppy host   makes mistakes (agent-driven: peeks at the answer before
                the guess; graph-driven: tries to skip to the wrong image),
                misjudges round 3, shows the wrong score, and tries to
                show a fifth image after the game is complete. Should
                lose points and FAIL the critical invariants.

Needs the optional extras:
    pip install -r requirements.txt -r requirements-frameworks.txt
Run from the project root:
    python testing/smoke_test_guess_game.py
"""

import json
import logging
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_router import ModelRouter  # noqa: E402
from workflows.guess_game.evaluator import GuessGameEvaluator  # noqa: E402
from workflows.guess_game.game import load_task  # noqa: E402
from workflows.guess_game.runner_autogen import GuessGameAutoGenRunner  # noqa: E402
from workflows.guess_game.runner_langgraph import GuessGameLangGraphRunner  # noqa: E402
from workflows.guess_game.runner_autogen_graph import GuessGameAutoGenGraphRunner  # noqa: E402
from workflows.guess_game.runner_langgraph_agent import GuessGameLangGraphAgentRunner  # noqa: E402

TASKS = ["benchmarks/guess_game/guess_the_person.yaml",
         "benchmarks/guess_game/guess_the_person_normalization.yaml"]
IDS = ["img-001", "img-002", "img-003", "img-004"]
INTRO = {"title": "Guess the Person",
         "rules": ["Four rounds, one picture each.", "One guess per image.",
                   "One point per correct guess; final score at the end."],
         "total_rounds": 4}
TRUE_MARKS = ["✅", "❌", "✅", "✅"]
SLOPPY_MARKS = ["✅", "❌", "❌", "✅"]   # wrongly rejects round 3


def feedback(mark):
    return ("text", f"{mark} " + ("Correct, well done!" if mark == "✅"
                                  else "Not this time, try again next round."))


# --- scripted steps: ("tool", name, args), ("text", str), or
#     ("both", text, name, args) for text sent together with a tool call ------
#
# Graph-driven runners (langgraph, autogen-graph) make one model call per
# step; agent-driven runners (autogen, langgraph-agent) chain steps inside a
# turn. The two scripts below cover each pattern, and each script is used
# by both frameworks of that pattern.

def graph_script(sloppy: bool):
    steps = [("tool", "show_intro_card", INTRO)]
    for r, mark in enumerate(SLOPPY_MARKS if sloppy else TRUE_MARKS, start=1):
        if sloppy and r == 2:
            steps.append(("tool", "show_image", {"image_id": "img-003"}))  # skips ahead: rejected, retried
        ask = "Who do you think this is?"
        if sloppy and r == 2:
            # Re-judges round 1 while asking, contradicting the ✅ it gave:
            # the pattern seen in real AutoGen GraphFlow runs.
            ask = "❌ Actually, that last one wasn't Ada Lovelace. Who do you think this is?"
        steps += [("tool", "show_image", {"image_id": IDS[r - 1]}), ("text", ask),
                  ("tool", "reveal_answer", {"image_id": IDS[r - 1]}), feedback(mark)]
    steps.append(("tool", "show_results_card", {"score": 2 if sloppy else 3, "total": 4}))
    if sloppy:
        steps.append(("tool", "show_image", {"image_id": "img-005"}))  # restarts after completion
    return steps + [("text", "Glad you enjoyed it! The game is over, but you can start a new one anytime.")]


def agent_script(sloppy: bool):
    # Like GPT-4o often does, the agent sends its feedback in the SAME
    # response as the next show_image call ("both"); that text must still
    # reach the player as feedback before the image is shown.
    marks = SLOPPY_MARKS if sloppy else TRUE_MARKS
    steps = [("tool", "show_intro_card", INTRO),
             ("tool", "show_image", {"image_id": IDS[0]}), ("text", "Who do you think this is?")]
    for r, mark in enumerate(marks, start=1):
        steps.append(("tool", "reveal_answer", {"image_id": IDS[r - 1]}))
        text = feedback(mark)[1]
        if sloppy and r == 2:
            text += " Here's the next one: ![Image 3](img-003)"   # image pasted into chat
        if r < len(marks):
            steps += [("both", text, "show_image", {"image_id": IDS[r]}),
                      ("text", "Who do you think this is?")]
        else:
            steps.append(("text", text))
    if sloppy:
        # Peeks at the answer before the guess in round 1: rejected.
        steps.insert(2, ("tool", "reveal_answer", {"image_id": "img-001"}))
    steps += [("tool", "show_results_card", {"score": 2 if sloppy else 3, "total": 4}),
              ("text", "Thanks for playing!")]
    if sloppy:
        steps.append(("tool", "show_image", {"image_id": "img-005"}))  # restarts after completion
    return steps + [("text", "Glad you enjoyed it! The game is over, but you can start a new one anytime.")]


# --- fake models -----------------------------------------------------------------

def fake_langchain_model(steps):
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class ScriptedChatModel(BaseChatModel):
        script: list
        cursor: list = [0]

        @property
        def _llm_type(self):
            return "scripted"

        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            i = min(self.cursor[0], len(self.script) - 1)
            self.cursor[0] += 1
            kind, *rest = self.script[i]
            usage = {"input_tokens": 300, "output_tokens": 40, "total_tokens": 340}
            if kind == "tool":
                msg = AIMessage(content="", usage_metadata=usage,
                                tool_calls=[{"name": rest[0], "args": rest[1], "id": f"call_{i}"}])
            elif kind == "both":
                msg = AIMessage(content=rest[0], usage_metadata=usage,
                                tool_calls=[{"name": rest[1], "args": rest[2], "id": f"call_{i}"}])
            else:
                msg = AIMessage(content=rest[0], usage_metadata=usage)
            return ChatResult(generations=[ChatGeneration(message=msg)])

    return ScriptedChatModel(script=steps, cursor=[0])


def fake_autogen_client(steps):
    from autogen_core import FunctionCall
    from autogen_core.models import CreateResult, RequestUsage
    from autogen_ext.models.replay import ReplayChatCompletionClient

    results = []
    for i, (kind, *rest) in enumerate(steps):
        usage = RequestUsage(prompt_tokens=300, completion_tokens=40)
        if kind == "tool":
            content = [FunctionCall(id=f"call_{i}", name=rest[0], arguments=json.dumps(rest[1]))]
            results.append(CreateResult(finish_reason="function_calls", content=content,
                                        usage=usage, cached=False))
        elif kind == "both":   # text + tool call: AutoGen puts the text in `thought`
            content = [FunctionCall(id=f"call_{i}", name=rest[1], arguments=json.dumps(rest[2]))]
            results.append(CreateResult(finish_reason="function_calls", content=content,
                                        thought=rest[0], usage=usage, cached=False))
        else:
            results.append(CreateResult(finish_reason="stop", content=rest[0], usage=usage, cached=False))
    return ReplayChatCompletionClient(results, model_info={
        "vision": False, "function_calling": True, "json_output": False,
        "family": "unknown", "structured_output": False})


def main():
    logging.disable(logging.WARNING)  # the replay client warns on every call
    router = ModelRouter([{"id": "fake", "provider": "openai", "api_key": "fake"}])
    evaluator = GuessGameEvaluator()
    checked = ("state_and_flow", "tool_use", "ui_control", "answer_handling",
               "response_quality", "critical_invariants")

    for task_path in TASKS:
        task = load_task(task_path)
        hosts = ("perfect", "sloppy") if task_path == TASKS[0] else ("perfect",)
        batch = []
        for host in hosts:
            sloppy = host == "sloppy"
            for fw, module, factory, fake, script, runner in [
                ("autogen", "runner_autogen", "make_autogen_client",
                 fake_autogen_client, agent_script, GuessGameAutoGenRunner),
                ("autogen-graph", "runner_autogen_graph", "make_autogen_client",
                 fake_autogen_client, graph_script, GuessGameAutoGenGraphRunner),
                ("langgraph-agent", "runner_langgraph_agent", "make_langchain_model",
                 fake_langchain_model, agent_script, GuessGameLangGraphAgentRunner),
                ("langgraph", "runner_langgraph", "make_langchain_model",
                 fake_langchain_model, graph_script, GuessGameLangGraphRunner),
            ]:
                with patch(f"workflows.guess_game.{module}.{factory}",
                           lambda *_, f=fake, s=script: f(s(sloppy))):
                    batch.append((fw, host, runner(router).run("fake", task)))

        print(f"\n=== {task.task_id} ===")
        for fw, host, tr, sr in evaluator.score_all(batch, task):
            print(f"\n{host} host on {fw} [{tr.extra.get('pattern')}]  (composite {sr.composite_score})")
            for k in checked:
                print(f"  {k:20s} {sr.dimensions[k]['score']:5.1f}  {sr.dimensions[k]['reason']}")
            scores = {k: sr.dimensions[k]["score"] for k in checked}
            if host == "perfect":
                assert all(v == 100.0 for v in scores.values()), (fw, sr.dimensions)
            else:
                assert scores["critical_invariants"] == 0.0, (fw, sr.dimensions)
                for k in ("state_and_flow", "tool_use", "answer_handling"):
                    assert scores[k] < 100, (fw, k, sr.dimensions)
                if "agent" in (tr.extra.get("pattern") or ""):
                    assert scores["ui_control"] < 100, (fw, "image pasted into chat", sr.dimensions)
                else:
                    assert "contradicting" in sr.dimensions["critical_invariants"]["reason"], fw

    print("\nGuess-the-person smoke test passed (real AutoGen + LangGraph, scripted models).")


if __name__ == "__main__":
    main()
