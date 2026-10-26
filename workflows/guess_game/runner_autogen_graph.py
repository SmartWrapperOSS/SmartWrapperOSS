# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

"""
workflows/guess_game/runner_autogen_graph.py   (--frameworks autogen-graph)

Runs guess_the_person on the REAL AutoGen package with GRAPH-DRIVEN FLOW,
using AutoGen's GraphFlow (a directed graph of agents). This is the
AutoGen counterpart of runner_langgraph.py, so the benchmark can separate
"which framework" from "which pattern":

                   agent-driven              graph-driven
    AutoGen        autogen                   autogen-graph   <- this file
    LangGraph      langgraph-agent           langgraph

One AssistantAgent per game phase, each holding ONLY the tool for its step
(all sharing one model client), plus the scripted player:

    intro       show_intro_card
    image       show_image   (still has to pick the right image id)
    ask         no tools (chat)
    player      scripted guess / post-game message
    judge       reveal_answer, then chat feedback
    results     show_results_card
    post_game   ALL tools, so "don't restart the game" is a real test
    game_over   leaf node (no model call); GraphFlow requires one

Every edge is conditional on the GameMachine's phase, so after any agent
speaks, the graph hands the turn to whichever agent the game needs next
(the same agent again if it didn't do its step). Agents see the shared
conversation. Unlike the LangGraph node instructions, each agent's role is
fixed in its system message, which is how GraphFlow agents are normally
set up; the current round number comes from the conversation.
"""

import asyncio
import time

from core.framework_clients import make_autogen_client
from core.model_router import ModelRouter
from core.types import TaskResult
from workflows.base import Workflow
from workflows.guess_game.game import (
    CORRECT_MARK, WRONG_MARK, GameMachine, GuessGameTask, Phase,
    host_system_prompt, make_card_tools,
)
from workflows.guess_game.runner_autogen import (
    _chat_delivering_client, _make_player_agent, _version,
)

MESSAGES_PER_ROUND = 5
MESSAGE_SLACK = 14

ROLES = {
    "intro": ("show_intro_card", 1, "Your job: show the introduction card at the start of the game."),
    "image": ("show_image", 1, "Your job: show the image for the next round with show_image."),
    "ask": (None, 1, "Your job: ask the player in chat who they think is in the image just shown."),
    "judge": ("reveal_answer", 2, f"Your job: the player has just guessed. Check it with "
                                  f"reveal_answer, then give feedback in chat starting with "
                                  f"{CORRECT_MARK} or {WRONG_MARK}."),
    "results": ("show_results_card", 1, "Your job: after the last round, show the final score "
                                         "in the results card."),
    "post_game": ("ALL", 2, "Your job: the game is complete. Reply to the player in chat."),
}

PHASE_TO_NODE = {
    Phase.INTRO: "intro", Phase.SHOW_IMAGE: "image", Phase.ASK: "ask",
    Phase.AWAIT_GUESS: "player", Phase.JUDGE: "judge", Phase.RESULTS: "results",
    Phase.AWAIT_POST_GAME: "player", Phase.POST_GAME: "post_game", Phase.DONE: "game_over",
}


def _make_game_over_agent():
    """GraphFlow requires a leaf node; this one just closes the graph once
    the game is DONE. It makes no model call."""
    from autogen_agentchat.agents import BaseChatAgent
    from autogen_agentchat.base import Response
    from autogen_agentchat.messages import TextMessage

    class GameOver(BaseChatAgent):
        def __init__(self):
            super().__init__("game_over", description="Ends the game graph.")

        @property
        def produced_message_types(self):
            return (TextMessage,)

        async def on_messages(self, messages, cancellation_token) -> Response:
            return Response(chat_message=TextMessage(content="(game over)", source=self.name))

        async def on_reset(self, cancellation_token) -> None:
            pass

    return GameOver()


class GuessGameAutoGenGraphRunner(Workflow):
    def __init__(self, router: ModelRouter):
        self.router = router

    def run(self, model_id: str, task_input: GuessGameTask) -> TaskResult:
        return asyncio.run(self._run(model_id, task_input))

    async def _run(self, model_id: str, task: GuessGameTask) -> TaskResult:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.conditions import FunctionalTermination, MaxMessageTermination
        from autogen_agentchat.messages import TextMessage
        from autogen_agentchat.teams import DiGraphBuilder, GraphFlow

        config = self.router.configs[model_id]
        client = make_autogen_client(model_id, config)
        machine = GameMachine(task)
        tools = {f.__name__: f for f in make_card_tools(machine)}
        base_prompt = host_system_prompt(task)

        # Host chat text has to reach the engine before the graph picks the
        # next speaker. Edge conditions see each message, so they feed it in
        # (once per message id, however many edges evaluate it).
        seen = set()

        def feed(msg) -> None:
            if msg.id in seen:
                return
            seen.add(msg.id)
            if msg.source in ROLES and isinstance(msg, TextMessage):
                machine.host_said(msg.content)

        delivering = _chat_delivering_client(client, machine)
        agents = {}
        for name, (tool, iterations, role) in ROLES.items():
            agent_tools = (list(tools.values()) if tool == "ALL"
                           else [tools[tool]] if tool else [])
            agents[name] = AssistantAgent(
                name, model_client=delivering, tools=agent_tools or None,
                system_message=f"{base_prompt}\n\n{role}",
                max_tool_iterations=iterations, reflect_on_tool_use=False,
            )
        agents["player"] = _make_player_agent(machine, feed_host_message=feed)

        agents["game_over"] = _make_game_over_agent()

        builder = DiGraphBuilder()
        for a in agents.values():
            builder.add_node(a)
        for src in agents:
            if src == "game_over":
                continue          # the leaf: no outgoing edges
            for dst in agents:
                def condition(msg, dst=dst):
                    feed(msg)
                    return PHASE_TO_NODE[machine.phase] == dst
                # "any": a node runs as soon as ONE incoming edge fires
                # (GraphFlow's default "all" would wait for every parent).
                builder.add_edge(agents[src], agents[dst], condition=condition,
                                 activation_condition="any")
        builder.set_entry_point(agents["intro"])

        cap = len(task.rounds) * MESSAGES_PER_ROUND + MESSAGE_SLACK
        termination = FunctionalTermination(lambda _m: machine.done) | MaxMessageTermination(cap)
        team = GraphFlow(list(agents.values()), graph=builder.build(),
                         termination_condition=termination)

        stop_reason = "finished"
        model_calls = 0
        start = time.time()
        try:
            result = await team.run(task=task.opening_message)
            model_calls = sum(1 for m in result.messages
                              if m.source in ROLES and getattr(m, "models_usage", None))
            if not machine.done:
                stop_reason = f"stopped before finishing ({result.stop_reason})"
        except Exception as e:  # a framework/provider error ends this run, scored as-is
            stop_reason = f"error: {type(e).__name__}: {str(e).splitlines()[0][:200]}"
        latency_ms = (time.time() - start) * 1000

        usage = client.total_usage()
        try:
            await client.close()
        except Exception:
            pass

        return TaskResult(
            output=stop_reason,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            latency_ms=latency_ms,
            cost_usd=self.router._calc_cost(model_id, usage.prompt_tokens,
                                            usage.completion_tokens, config),
            extra={"events": machine.events, "stop_reason": stop_reason,
                   "final_phase": machine.phase.value, "model_calls": model_calls,
                   "pattern": "graph-driven",
                   "framework_version": _version("autogen-agentchat")},
        )
