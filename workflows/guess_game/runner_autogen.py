# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

"""
workflows/guess_game/runner_autogen.py

Runs the guessing game on the REAL AutoGen AgentChat package (0.7.x), in
AutoGen's typical pattern: AGENT-DRIVEN FLOW.

    RoundRobinGroupChat([host, player])
        host   = AssistantAgent with all four game tools (native function
                 calling via AutoGen's model client) and the game rules as
                 its system message. It decides on its own what to do next.
        player = a scripted agent backed by the shared GameMachine.

There is no per-step instruction: the host only has the rules and the
conversation, so this measures how well the agent keeps to the flow by
itself. Each host turn may chain up to MAX_TOOL_ITERATIONS tool calls
before replying in chat (e.g. show an image card, then ask "who is it?").

As in AutoGen generally, text the model writes ALONGSIDE a tool call is a
ThoughtEvent, not a chat message, so only a turn's final text reaches the
player. The LangGraph runner treats it the same way.
"""

import asyncio
import time
from typing import Sequence

from core.framework_clients import make_autogen_client
from core.model_router import ModelRouter
from core.types import TaskResult
from workflows.base import Workflow
from workflows.guess_game.game import (
    GameMachine, GuessGameTask, host_system_prompt, make_card_tools,
)

MAX_TOOL_ITERATIONS = 3
# Host + player messages: ~2 for the intro, ~4 per round, ~4 for the
# results and the post-game exchange, plus slack. A host that loses the
# plot still stops.
MESSAGES_PER_ROUND = 4
MESSAGE_SLACK = 10


def _make_player_agent(machine: GameMachine, feed_host_message=None):
    from autogen_agentchat.agents import BaseChatAgent
    from autogen_agentchat.base import Response
    from autogen_agentchat.messages import TextMessage

    class ScriptedPlayer(BaseChatAgent):
        """Feeds the host's chat messages into the GameMachine and answers
        with the scripted guess when one is expected."""

        def __init__(self):
            super().__init__("player", description="The person playing the game.")

        @property
        def produced_message_types(self):
            return (TextMessage,)

        async def on_messages(self, messages: Sequence, cancellation_token) -> Response:
            for m in messages:
                # Only the host's chat text reaches the player; tool calls
                # and their results are events, not chat.
                if feed_host_message:
                    feed_host_message(m)
                elif m.source == "host" and isinstance(m, TextMessage):
                    machine.host_said(m.content)
            guess = machine.player_reply()
            return Response(chat_message=TextMessage(content=guess or "Okay!", source=self.name))

        async def on_reset(self, cancellation_token) -> None:
            pass

    return ScriptedPlayer()


class GuessGameAutoGenRunner(Workflow):
    def __init__(self, router: ModelRouter):
        self.router = router

    def run(self, model_id: str, task_input: GuessGameTask) -> TaskResult:
        return asyncio.run(self._run(model_id, task_input))

    async def _run(self, model_id: str, task: GuessGameTask) -> TaskResult:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.conditions import FunctionalTermination, MaxMessageTermination
        from autogen_agentchat.messages import TextMessage
        from autogen_agentchat.teams import RoundRobinGroupChat

        config = self.router.configs[model_id]
        client = make_autogen_client(model_id, config)
        machine = GameMachine(task)

        host = AssistantAgent(
            "host",
            model_client=client,
            tools=make_card_tools(machine),
            system_message=host_system_prompt(task),
            max_tool_iterations=MAX_TOOL_ITERATIONS,
            reflect_on_tool_use=False,
        )
        player = _make_player_agent(machine)

        cap = len(task.rounds) * MESSAGES_PER_ROUND + MESSAGE_SLACK
        termination = FunctionalTermination(lambda _msgs: machine.done) | MaxMessageTermination(cap)
        team = RoundRobinGroupChat([host, player], termination_condition=termination)

        stop_reason = "finished"
        model_calls = 0
        start = time.time()
        try:
            result = await team.run(task=task.opening_message)
            model_calls = sum(1 for m in result.messages
                              if m.source == "host" and getattr(m, "models_usage", None))
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
                   "pattern": "agent-driven",
                   "framework_version": _version("autogen-agentchat")},
        )


def _version(pkg: str) -> str:
    try:
        from importlib.metadata import version
        return version(pkg)
    except Exception:
        return "unknown"
