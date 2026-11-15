# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

"""
workflows/tool_use/runner_ensemble.py

Runs the tool-calling task using a parallel Ensemble / Swarm orchestration
pattern with Parallel Tool-Proposal Voting:

  Each turn (up to MAX_TURNS):
    1. Phase 1 ("Swarm"): N specialist agents inspect the task, available tools,
       and conversation history in parallel via ThreadPoolExecutor, each
       proposing a JSON decision (`call_tool` or `final_answer`).
    2. Phase 2 ("Consensus"):
       - If valid proposals have a unanimous or strict-majority exact match on
         (action, tool, args), that consensus decision is selected directly.
       - Otherwise, a Consensus Synthesizer LLM call arbitrates across all N
         proposals to select the safest, most accurate next action.
    3. Execution: Only ONE agreed-upon tool call is executed against
       `ToolProvider` per turn, preventing duplicate side-effects while
       benefiting from multi-agent verification.

Wall-clock latency (`latency_ms`) tracks the critical path per turn
(`max(swarm_latencies) + synthesizer_latency`), while token counts and
`cost_usd` sum across all LLM calls.
"""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time
from typing import List, Optional

from core.model_router import ModelResponse, ModelRouter
from core.types import TaskResult
from workflows.base import Workflow
from workflows.tool_use.tasks import ToolUseTask
from workflows.tool_use.tool_provider import build_tool_provider

MAX_TURNS = 6

DEFAULT_TOOL_PERSONAS = [
    "Task Decomposition & Sequencing Specialist",
    "Tool-Schema & Argument Precision Auditor",
    "Precondition & Dependency Verification Specialist",
    "Edge-Case & Error-Recovery Analyst",
    "Minimal-Step Efficiency Optimizer",
]

SWARM_TOOL_PROMPT = """You are Agent #{idx} ({persona}) in a parallel tool-calling ensemble.
Propose the single best next action for the task below. Pay special attention to your specialty ({persona}).

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

SYNTHESIZER_TOOL_PROMPT = """You are the Lead Consensus Arbitrator for a {num_agents}-agent tool-calling ensemble.
Below are {num_agents} candidate next-step proposals for the current turn.
Select or merge the most accurate, schema-valid next action that respects tool preconditions and avoids redundant calls.

TASK:
{prompt}

AVAILABLE TOOLS:
{tool_descriptions}

CONVERSATION SO FAR:
{history}

CANDIDATE PROPOSALS:
{candidates}

Respond with ONLY valid JSON in ONE of these two forms:
{{"action": "call_tool", "tool": "<tool_name>", "args": {{...}}}}
or
{{"action": "final_answer", "answer": "<your final answer to the user>"}}
"""


class ToolUseEnsembleRunner(Workflow):
    def __init__(self, router: ModelRouter, num_agents: int = 5, max_workers: int = 20):
        self.router = router
        self.num_agents = max(1, num_agents)
        self.max_workers = max_workers

    def _get_personas(self) -> List[str]:
        return [
            DEFAULT_TOOL_PERSONAS[i % len(DEFAULT_TOOL_PERSONAS)]
            for i in range(self.num_agents)
        ]

    def _call_with_retry(self, model_id: str, prompt: str, retries: int = 3) -> ModelResponse:
        last_err = None
        for attempt in range(retries):
            try:
                return self.router.call(model_id, prompt)
            except Exception as e:
                last_err = e
                if attempt < retries - 1:
                    time.sleep(0.6 * (attempt + 1))
        raise last_err


    def run(self, model_id: str, task_input: ToolUseTask) -> TaskResult:
        task = task_input
        personas = self._get_personas()

        with build_tool_provider(task) as provider:
            tool_descriptions = self._describe_tools(provider)

            history: List[str] = []
            tool_calls: List[dict] = []
            total_input_tokens = 0
            total_output_tokens = 0
            total_latency_ms = 0.0
            total_cost_usd = 0.0
            final_answer = ""

            for _turn in range(MAX_TURNS):
                history_str = "\n".join(history) if history else "(nothing yet)"

                # Phase 1: Parallel Swarm Fan-Out
                swarm_responses: List[Optional[ModelResponse]] = [None] * self.num_agents
                with ThreadPoolExecutor(max_workers=min(self.max_workers, self.num_agents)) as pool:
                    future_to_idx = {
                        pool.submit(
                            self._call_with_retry,
                            model_id,
                            SWARM_TOOL_PROMPT.format(
                                idx=i + 1,
                                persona=persona,
                                prompt=task.prompt,
                                tool_descriptions=tool_descriptions,
                                history=history_str,
                            ),
                        ): i
                        for i, persona in enumerate(personas)
                    }
                    for fut in as_completed(future_to_idx):
                        idx = future_to_idx[fut]
                        swarm_responses[idx] = fut.result()

                valid_resps = [r for r in swarm_responses if r is not None]
                total_input_tokens += sum(r.input_tokens for r in valid_resps)
                total_output_tokens += sum(r.output_tokens for r in valid_resps)
                total_cost_usd += sum(r.cost_usd for r in valid_resps)
                turn_critical_latency = max((r.latency_ms for r in valid_resps), default=0.0)

                # Parse all candidate decisions
                parsed = []
                for r in valid_resps:
                    d = self._parse_decision(r.text)
                    if d is not None and d.get("action") in ("call_tool", "final_answer"):
                        parsed.append(d)

                decision = self._majority_decision(parsed)
                if decision is None:
                    # Phase 2: Consensus Arbitrator when proposals disagree or fail to parse
                    candidates_block = "\n\n".join(
                        f"--- Proposal #{i + 1} ({personas[i]}) ---\n{r.text if r else '(no response)'}"
                        for i, r in enumerate(swarm_responses)
                    )
                    synth_prompt = SYNTHESIZER_TOOL_PROMPT.format(
                        num_agents=self.num_agents,
                        prompt=task.prompt,
                        tool_descriptions=tool_descriptions,
                        history=history_str,
                        candidates=candidates_block,
                    )
                    synth_resp = self._call_with_retry(model_id, synth_prompt)
                    total_input_tokens += synth_resp.input_tokens
                    total_output_tokens += synth_resp.output_tokens
                    total_cost_usd += synth_resp.cost_usd
                    turn_critical_latency += synth_resp.latency_ms
                    decision = self._parse_decision(synth_resp.text)

                total_latency_ms += turn_critical_latency

                if decision is None:
                    history.append("(unparseable consensus response, stopping)")
                    break

                if decision.get("action") == "final_answer":
                    final_answer = decision.get("answer", "")
                    history.append(f"Assistant gave final answer: {final_answer}")
                    break

                elif decision.get("action") == "call_tool":
                    tool_name = decision.get("tool")
                    args = decision.get("args", {})
                    result, exec_latency_ms = self._execute_tool(provider, tool_name, args)
                    total_latency_ms += exec_latency_ms

                    tool_calls.append({
                        "tool_name": tool_name,
                        "arguments": args,
                        "result": result,
                        "timestamp": time.time(),
                    })
                    history.append(f"Assistant called {tool_name}({args}) -> {result}")
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

    def _majority_decision(self, parsed: List[dict]) -> Optional[dict]:
        """Return the consensus decision if a strict majority agrees on the exact action."""
        if not parsed:
            return None
        if len(parsed) == 1 and self.num_agents == 1:
            return parsed[0]

        def _key(d: dict) -> str:
            if d.get("action") == "call_tool":
                return json.dumps(
                    {"action": "call_tool", "tool": d.get("tool"), "args": d.get("args", {})},
                    sort_keys=True,
                )
            return json.dumps({"action": "final_answer"}, sort_keys=True)

        counts = Counter(_key(d) for d in parsed)
        top_key, top_count = counts.most_common(1)[0]
        if top_count > self.num_agents / 2:
            for d in parsed:
                if _key(d) == top_key:
                    return d
        return None

    def _describe_tools(self, provider) -> str:
        return "\n".join(f"- {s.name}: {s.description}" for s in provider.list_tools())

    def _parse_decision(self, raw_text: str):
        text = raw_text.strip()
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                p = part.strip()
                if p.startswith("json"):
                    p = p[4:].strip()
                if p.startswith("{"):
                    text = p
                    break
        try:
            return json.loads(text)
        except (json.JSONDecodeError, IndexError):
            return None

    def _execute_tool(self, provider, tool_name: str, args: dict):
        start = time.time()
        result = provider.call_tool(tool_name, args)
        latency_ms = (time.time() - start) * 1000
        return result, latency_ms
