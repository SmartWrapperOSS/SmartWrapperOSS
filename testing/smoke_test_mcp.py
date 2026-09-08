# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

"""
testing/smoke_test_mcp.py

Same idea as the smoke_test.py in TESTING.md section 2 — a scripted fake
ModelRouter, no real API keys, no real API cost — but this one also
spawns the real (pinned, local) testing/mcp_fixture_server.py subprocess,
so it verifies the actual MCP wire protocol, not just a mocked client.

Run from the project root:
    python testing/smoke_test_mcp.py

Expected output: a comparison table with one row (autogen / fake-model),
all four mechanical dimensions at 100, and "MCP smoke test passed."
"""

import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.getcwd())  # run from project root

from core.model_router import ModelRouter, ModelResponse
from core.types import EvalResult
from workflows.tool_use.runner_autogen import ToolUseAutoGenRunner
from workflows.tool_use.evaluator import ToolUseEvaluator
from workflows.tool_use.tasks import load_task
from output.formatter import print_table

SCRIPTED_RESPONSES = [
    '{"action": "call_tool", "tool": "get_weather", "args": {"location": "Boston", "date": "tomorrow"}}',
    '{"action": "call_tool", "tool": "get_calendar_event", "args": {"time": "3pm"}}',
    '{"action": "call_tool", "tool": "update_calendar_event", "args": {"new_time": "5pm"}}',
    '{"action": "final_answer", "answer": "Moved your 3pm meeting to 5pm due to rain."}',
]
call_count = {"n": 0}


def fake_call(self, model_id, prompt):
    idx = min(call_count["n"], len(SCRIPTED_RESPONSES) - 1)
    call_count["n"] += 1
    return ModelResponse(
        model_id=model_id, text=SCRIPTED_RESPONSES[idx],
        input_tokens=100, output_tokens=20, latency_ms=250.0, cost_usd=0.001,
    )


router = ModelRouter([{"id": "fake-model", "provider": "openai", "api_key": "fake"}])
task = load_task("benchmarks/tool_use/weather_and_calendar_mcp.yaml")

with patch.object(ModelRouter, "call", fake_call):
    result = ToolUseAutoGenRunner(router).run("fake-model", task)

print("Tool calls made:", [c["tool_name"] for c in result.extra["tool_calls"]])

evaluator = ToolUseEvaluator()
scored = evaluator.score_all([("autogen", "fake-model", result)], task)
results = [EvalResult(framework=fw, model_id=m, task_result=tr, score_result=sr)
           for fw, m, tr, sr in scored]

print_table(results)
print("MCP smoke test passed.")
