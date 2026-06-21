# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

"""
workflows/tool_use/runner_autogen.py

Runs a tool-calling agent loop in an AutoGen-style pattern: a single agent
repeatedly asks the model "what should I do next?", the model responds
with either a tool call or a final answer, and the agent executes tool
calls one at a time until the model says it's done.

This is a SIMPLIFIED simulation of AutoGen's tool-calling pattern, not a
direct dependency on the `autogen` package — it's built this way so the
project has no hard dependency on either framework's exact API, and so the
comparison logic (how many turns, which tools, in what order) is fully
visible and easy to follow rather than hidden inside a library.

Both this file and runner_langgraph.py produce the SAME normalized output
shape (a list of ToolCall records inside TaskResult.extra), which is what
lets the ToolUseEvaluator compare them fairly — see core/types.py for why
that normalization matters.
"""

import json
import time

from core.types import TaskResult
from core.model_router import ModelRouter
from workflows.base import Workflow
from workflows.tool_use.tasks import ToolUseTask, TOOL_REGISTRY

MAX_TURNS = 6  # safety cap so a confused agent can't loop forever

AGENT_PROMPT = """You are an assistant that can call tools to complete tasks.

TASK:
{prompt}

AVAILABLE TOOLS:
{tool_descriptions}

CONVERSATION SO FAR:
{history}

Respond with ONLY valid JSON in ONE of these two forms:

To call a tool:
{{"action": "call_tool", "tool": "<tool_name>", "args": {{...}}}}

When the task is fully complete:
{{"action": "final_answer", "answer": "<your final answer to the user>"}}
"""


class ToolUseAutoGenRunner(Workflow):
    def __init__(self, router: ModelRouter, agent_prompt: str = AGENT_PROMPT):
        """
        `agent_prompt` can be overridden — e.g. from config/config.yaml's
        `prompts:` section — to test how instructions (tone, strictness,
        examples) affect tool-selection accuracy. Defaults to the built-in
        text above if not overridden.

        Must contain the same {prompt}, {tool_descriptions}, and {history}
        placeholders as the default above, since those are filled in each
        turn.
        """
        self.router = router
        self.agent_prompt = agent_prompt

    def run(self, model_id: str, task_input: ToolUseTask) -> TaskResult:
        task = task_input
        tool_descriptions = self._describe_tools(task.available_tools)

        history = []          # human-readable log, shown back to the model each turn
        tool_calls = []       # structured trace, used for scoring later
        total_input_tokens = 0
        total_output_tokens = 0
        total_latency_ms = 0.0
        total_cost_usd = 0.0
        final_answer = ""

        for turn in range(MAX_TURNS):
            prompt = self.agent_prompt.format(
                prompt=task.prompt,
                tool_descriptions=tool_descriptions,
                history="\n".join(history) if history else "(nothing yet)",
            )

            response = self.router.call(model_id, prompt)
            total_input_tokens += response.input_tokens
            total_output_tokens += response.output_tokens
            total_latency_ms += response.latency_ms
            total_cost_usd += response.cost_usd

            decision = self._parse_decision(response.text)

            if decision is None:
                history.append("(unparseable response, stopping)")
                break

            if decision.get("action") == "final_answer":
                final_answer = decision.get("answer", "")
                history.append(f"Assistant gave final answer: {final_answer}")
                break

            elif decision.get("action") == "call_tool":
                tool_name = decision.get("tool")
                args = decision.get("args", {})
                result, latency_ms = self._execute_tool(tool_name, args)

                tool_calls.append({
                    "tool_name": tool_name,
                    "arguments": args,
                    "result": result,
                    "timestamp": time.time(),
                })
                history.append(
                    f"Assistant called {tool_name}({args}) -> {result}"
                )
            else:
                history.append("(unrecognized action, stopping)")
                break

        return TaskResult(
            output=final_answer,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            latency_ms=total_latency_ms,
            cost_usd=total_cost_usd,
            extra={"tool_calls": tool_calls},
        )

    def _describe_tools(self, tool_names: list) -> str:
        lines = []
        for name in tool_names:
            fn = TOOL_REGISTRY.get(name)
            doc = fn.__doc__.strip() if fn and fn.__doc__ else "(no description)"
            lines.append(f"- {name}: {doc}")
        return "\n".join(lines)

    def _parse_decision(self, raw_text: str):
        text = raw_text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        try:
            return json.loads(text)
        except (json.JSONDecodeError, IndexError):
            return None

    def _execute_tool(self, tool_name: str, args: dict):
        start = time.time()
        fn = TOOL_REGISTRY.get(tool_name)
        if fn is None:
            result = {"error": f"Unknown tool: {tool_name}"}
        else:
            try:
                result = fn(**args)
            except TypeError as e:
                result = {"error": f"Bad arguments: {e}"}
        latency_ms = (time.time() - start) * 1000
        return result, latency_ms
