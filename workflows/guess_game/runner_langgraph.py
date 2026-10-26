# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

"""
workflows/guess_game/runner_langgraph.py

Runs guess_the_person on the REAL LangGraph package (1.x), in LangGraph's
typical pattern: GRAPH-DRIVEN FLOW.

    START -> route -> intro | show_image | ask | player | judge | results
                      | post_game | END     (chosen from the game phase)

    intro       LLM node, binds ONLY show_intro_card
    show_image  LLM node, binds ONLY show_image (the model must still pick
                the right image id from the list; the graph doesn't tell it)
    ask         LLM node, no tools (chat)
    player      scripted node: the guess, or the post-game message
    judge       LLM node, binds ONLY reveal_answer, then gives feedback
    results     LLM node, binds ONLY show_results_card
    post_game   LLM node after completion, binds ALL tools, so the test of
                "don't restart the game" is real, not structural
    tools       LangGraph's prebuilt ToolNode; then routes back by phase

If a node doesn't do its step (e.g. chats instead of calling its tool, or
calls it wrongly and gets rejected), routing sends it back to the same
node; that retry costs a model call and counts toward the step limit.
Compare with runner_autogen.py, where one agent holds every tool and must
keep to the flow by itself. Same tools, prompt, player, and scoring.
"""

import time
from typing import Annotated, TypedDict

from core.framework_clients import make_langchain_model
from core.model_router import ModelRouter
from core.types import TaskResult
from workflows.base import Workflow
from workflows.guess_game.game import (
    CORRECT_MARK, WRONG_MARK, GameMachine, GuessGameTask, Phase,
    host_system_prompt, make_card_tools,
)

# Node executions: ~3 for the intro, ~8 per round, ~5 for results and the
# post-game reply, plus slack for retries.
STEPS_PER_ROUND = 8
STEP_SLACK = 16


class GuessGameLangGraphRunner(Workflow):
    def __init__(self, router: ModelRouter):
        self.router = router

    def run(self, model_id: str, task_input: GuessGameTask) -> TaskResult:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        from langchain_core.tools import StructuredTool
        from langgraph.errors import GraphRecursionError
        from langgraph.graph import END, START, StateGraph
        from langgraph.graph.message import add_messages
        from langgraph.prebuilt import ToolNode

        task = task_input
        config = self.router.configs[model_id]
        model = make_langchain_model(model_id, config)
        machine = GameMachine(task)
        tools = {f.__name__: StructuredTool.from_function(f) for f in make_card_tools(machine)}
        system = SystemMessage(host_system_prompt(task))
        usage = {"in": 0, "out": 0, "calls": 0}

        class State(TypedDict):
            messages: Annotated[list, add_messages]

        def step_instruction(phase: Phase) -> str:
            n, r = machine.total, machine.round
            return {
                Phase.INTRO: "Show the introduction card now.",
                Phase.SHOW_IMAGE: f"Start round {r} of {n}: show the next image.",
                Phase.ASK: "Ask the player in chat who they think is in the image.",
                Phase.JUDGE: (f"The player just guessed. Check it with reveal_answer if you haven't, "
                              f"then give feedback in chat starting with {CORRECT_MARK} or {WRONG_MARK}."),
                Phase.RESULTS: "All rounds are done. Show the final score in the results card.",
                Phase.POST_GAME: "The game is complete. Reply to the player in chat.",
            }[phase]

        def llm_node(phase: Phase, tool_names=()):
            bound = model.bind_tools([tools[t] for t in tool_names]) if tool_names else model

            def node(state: State):
                # The step instruction goes with this call only (it isn't
                # stored in the conversation) and is labeled as coming from
                # the game engine, not the player.
                msgs = [system, *state["messages"],
                        HumanMessage(f"[Game engine, not the player] {step_instruction(phase)}")]
                ai: AIMessage = bound.invoke(msgs)
                meta = getattr(ai, "usage_metadata", None) or {}
                usage["in"] += meta.get("input_tokens", 0)
                usage["out"] += meta.get("output_tokens", 0)
                usage["calls"] += 1
                if not ai.tool_calls:
                    machine.host_said(_text(ai.content))
                return {"messages": [ai]}
            return node

        def player_node(state: State):
            return {"messages": [HumanMessage(machine.player_reply() or "Okay!")]}

        def route(_state: State = None) -> str:
            return {Phase.INTRO: "intro", Phase.SHOW_IMAGE: "show_image", Phase.ASK: "ask",
                    Phase.AWAIT_GUESS: "player", Phase.JUDGE: "judge", Phase.RESULTS: "results",
                    Phase.AWAIT_POST_GAME: "player", Phase.POST_GAME: "post_game",
                    Phase.DONE: END}[machine.phase]

        def after_llm(state: State) -> str:
            last = state["messages"][-1]
            return "tools" if getattr(last, "tool_calls", None) else route()

        g = StateGraph(State)
        g.add_node("intro", llm_node(Phase.INTRO, ["show_intro_card"]))
        g.add_node("show_image", llm_node(Phase.SHOW_IMAGE, ["show_image"]))
        g.add_node("ask", llm_node(Phase.ASK))
        g.add_node("player", player_node)
        g.add_node("judge", llm_node(Phase.JUDGE, ["reveal_answer"]))
        g.add_node("results", llm_node(Phase.RESULTS, ["show_results_card"]))
        g.add_node("post_game", llm_node(Phase.POST_GAME, list(tools)))
        g.add_node("tools", ToolNode(list(tools.values())))

        targets = ["intro", "show_image", "ask", "player", "judge", "results", "post_game", END]
        g.add_conditional_edges(START, route, targets)
        for name in ("intro", "show_image", "ask", "judge", "results", "post_game"):
            g.add_conditional_edges(name, after_llm, targets + ["tools"])
        g.add_conditional_edges("tools", route, targets)
        g.add_conditional_edges("player", route, targets)
        graph = g.compile()

        stop_reason = "finished"
        limit = len(task.rounds) * STEPS_PER_ROUND + STEP_SLACK
        start = time.time()
        try:
            graph.invoke({"messages": [HumanMessage(task.opening_message)]},
                         {"recursion_limit": limit})
            if not machine.done:
                stop_reason = "stopped before finishing"
        except GraphRecursionError:
            stop_reason = f"stopped before finishing (step limit {limit} reached)"
        except Exception as e:  # a framework/provider error ends this run, scored as-is
            stop_reason = f"error: {type(e).__name__}: {str(e).splitlines()[0][:200]}"
        latency_ms = (time.time() - start) * 1000

        return TaskResult(
            output=stop_reason,
            input_tokens=usage["in"],
            output_tokens=usage["out"],
            latency_ms=latency_ms,
            cost_usd=self.router._calc_cost(model_id, usage["in"], usage["out"], config),
            extra={"events": machine.events, "stop_reason": stop_reason,
                   "final_phase": machine.phase.value, "model_calls": usage["calls"],
                   "pattern": "graph-driven",
                   "framework_version": _version("langgraph")},
        )


def _text(content) -> str:
    """AIMessage.content can be a string or a list of content blocks."""
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))


def _version(pkg: str) -> str:
    try:
        from importlib.metadata import version
        return version(pkg)
    except Exception:
        return "unknown"
