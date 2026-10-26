# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

"""
workflows/guess_game/runner_langgraph_agent.py   (--frameworks langgraph-agent)

Runs guess_the_person on the REAL LangGraph package with AGENT-DRIVEN
FLOW, using LangGraph's prebuilt `create_react_agent`: one agent with all
four game tools and the game rules as its prompt, looping model -> tools
-> model until it replies in chat. This is the LangGraph counterpart of
runner_autogen.py, so the benchmark can separate "which framework" from
"which pattern":

                   agent-driven              graph-driven
    AutoGen        autogen                   autogen-graph
    LangGraph      langgraph-agent  <- this  langgraph

Each player message is one invocation of the agent with the conversation
so far, which is the standard way to chat with a ReAct agent. There are no
per-step instructions: like the AutoGen agent, it has to keep to the flow
by itself.
"""

import time
import warnings

from core.framework_clients import make_langchain_model
from core.model_router import ModelRouter
from core.types import TaskResult
from workflows.base import Workflow
from workflows.guess_game.game import (
    GameMachine, GuessGameTask, host_system_prompt, make_card_tools,
)
from workflows.guess_game.runner_langgraph import _text, _version

# Host replies: ~1 for the intro, ~2 per round, ~2 at the end, plus slack.
REPLIES_PER_ROUND = 2
REPLY_SLACK = 6
STEPS_PER_REPLY = 12   # model/tool steps allowed inside one agent invocation


class GuessGameLangGraphAgentRunner(Workflow):
    def __init__(self, router: ModelRouter):
        self.router = router

    def run(self, model_id: str, task_input: GuessGameTask) -> TaskResult:
        from langchain_core.messages import AIMessage, HumanMessage
        from langgraph.errors import GraphRecursionError
        from langgraph.prebuilt import create_react_agent

        task = task_input
        config = self.router.configs[model_id]
        model = make_langchain_model(model_id, config)
        machine = GameMachine(task)

        def deliver_chat(state):
            # Runs after every model call and BEFORE any tool executes, so
            # text sent together with a tool call reaches the player first.
            last = state["messages"][-1]
            if isinstance(last, AIMessage):
                text = _text(last.content)
                if last.tool_calls:
                    if text.strip():
                        machine.host_said(text, with_tool_call=True)
                else:
                    machine.host_said(text)
            return {}

        with warnings.catch_warnings():
            # create_react_agent is deprecated in favor of langchain.agents
            # but still ships (and is pinned) in langgraph 1.x.
            warnings.simplefilter("ignore")
            agent = create_react_agent(model, make_card_tools(machine),
                                       prompt=host_system_prompt(task),
                                       post_model_hook=deliver_chat)

        usage = {"in": 0, "out": 0, "calls": 0}
        messages = [HumanMessage(task.opening_message)]
        stop_reason = "stopped before finishing (reply limit reached)"
        start = time.time()
        try:
            for _ in range(len(task.rounds) * REPLIES_PER_ROUND + REPLY_SLACK):
                before = len(messages)
                out = agent.invoke({"messages": messages},
                                   {"recursion_limit": STEPS_PER_REPLY})
                messages = out["messages"]
                for m in messages[before:]:
                    if isinstance(m, AIMessage):
                        meta = m.usage_metadata or {}
                        usage["in"] += meta.get("input_tokens", 0)
                        usage["out"] += meta.get("output_tokens", 0)
                        usage["calls"] += 1
                if machine.done:
                    stop_reason = "finished"
                    break
                messages = messages + [HumanMessage(machine.player_reply())]
        except GraphRecursionError:
            stop_reason = f"stopped before finishing (step limit {STEPS_PER_REPLY} in one reply)"
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
                   "pattern": "agent-driven",
                   "framework_version": _version("langgraph")},
        )
