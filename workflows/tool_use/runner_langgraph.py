# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

"""
workflows/tool_use/runner_langgraph.py

Runs the SAME tool-calling task as runner_autogen.py, but structured as an
explicit state graph instead of a conversational loop:

    decide_node -> (call_tool_node -> decide_node)*  -> done

The state (history, tool_calls, token totals) is passed explicitly between
node calls, mirroring LangGraph's style. Compare this file side-by-side
with runner_autogen.py: same task, same MAX_TURNS safety cap, same
TaskResult shape out the other end — the only real difference is HOW the
loop is structured, which is exactly the kind of difference this project
exists to make visible and measurable.
"""

import json
import time
from typing import TypedDict, List

from core.types import TaskResult
from core.model_router import ModelRouter
from workflows.base import Workflow
from workflows.tool_use.tasks import ToolUseTask
from workflows.tool_use.tool_provider import build_tool_provider

MAX_TURNS = 6

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


class GraphState(TypedDict):
    task: ToolUseTask
    model_id: str
    history: List[str]
    tool_calls: List[dict]
    final_answer: str
    done: bool
    total_input_tokens: int
    total_output_tokens: int
    total_latency_ms: float
    total_cost_usd: float
    agent_prompt: str
    tool_provider: object  # ToolProvider — set once in run(), read by both nodes


class ToolUseLangGraphRunner(Workflow):
    def __init__(self, router: ModelRouter, agent_prompt: str = AGENT_PROMPT):
        """
        `agent_prompt` can be overridden — e.g. from config/config.yaml's
        `prompts:` section. Defaults to the built-in text above. Must
        contain the same {prompt}, {tool_descriptions}, and {history}
        placeholders, since those are filled in each turn.
        """
        self.router = router
        self.agent_prompt = agent_prompt

    def run(self, model_id: str, task_input: ToolUseTask) -> TaskResult:
        # A fresh provider per call: main.py runs multiple (framework,
        # model, run_index) combinations concurrently via a
        # ThreadPoolExecutor, so each call gets its own subprocess/session
        # rather than sharing one across threads.
        with build_tool_provider(task_input) as provider:
            state: GraphState = {
                "task": task_input,
                "model_id": model_id,
                "history": [],
                "tool_calls": [],
                "final_answer": "",
                "done": False,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_latency_ms": 0.0,
                "total_cost_usd": 0.0,
                "agent_prompt": self.agent_prompt,
                "tool_provider": provider,
            }

            # The graph loop: decide_node runs, and if it decided to call a
            # tool (not done yet), call_tool_node runs before looping back.
            for turn in range(MAX_TURNS):
                state = self._decide_node(state)
                if state["done"]:
                    break
                state = self._call_tool_node(state)

        return TaskResult(
            output=state["final_answer"],
            input_tokens=state["total_input_tokens"],
            output_tokens=state["total_output_tokens"],
            latency_ms=state["total_latency_ms"],
            cost_usd=state["total_cost_usd"],
            extra={"tool_calls": state["tool_calls"]},
        )

    def _decide_node(self, state: GraphState) -> GraphState:
        """Node: ask the model what to do next, given history so far."""
        task = state["task"]
        prompt = state["agent_prompt"].format(
            prompt=task.prompt,
            tool_descriptions=self._describe_tools(state["tool_provider"]),
            history="\n".join(state["history"]) if state["history"] else "(nothing yet)",
        )

        response = self.router.call(state["model_id"], prompt)
        state["total_input_tokens"] += response.input_tokens
        state["total_output_tokens"] += response.output_tokens
        state["total_latency_ms"] += response.latency_ms
        state["total_cost_usd"] += response.cost_usd

        decision = self._parse_decision(response.text)

        if decision is None:
            state["history"].append("(unparseable response, stopping)")
            state["done"] = True
            return state

        if decision.get("action") == "final_answer":
            state["final_answer"] = decision.get("answer", "")
            state["history"].append(f"Assistant gave final answer: {state['final_answer']}")
            state["done"] = True
            return state

        if decision.get("action") == "call_tool":
            # Stash the pending call; call_tool_node executes it next.
            state["_pending_call"] = decision
            state["done"] = False
            return state

        state["history"].append("(unrecognized action, stopping)")
        state["done"] = True
        return state

    def _call_tool_node(self, state: GraphState) -> GraphState:
        """Node: execute whatever tool decide_node just decided to call."""
        decision = state.get("_pending_call")
        if not decision:
            return state

        tool_name = decision.get("tool")
        args = decision.get("args", {})
        result = self._execute_tool(state["tool_provider"], tool_name, args)

        state["tool_calls"].append({
            "tool_name": tool_name,
            "arguments": args,
            "result": result,
            "timestamp": time.time(),
        })
        state["history"].append(f"Assistant called {tool_name}({args}) -> {result}")
        return state

    def _describe_tools(self, provider) -> str:
        return "\n".join(f"- {s.name}: {s.description}" for s in provider.list_tools())

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

    def _execute_tool(self, provider, tool_name: str, args: dict) -> dict:
        return provider.call_tool(tool_name, args)
