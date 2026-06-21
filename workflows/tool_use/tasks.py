# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

"""
workflows/tool_use/tasks.py

Defines what a "tool-use benchmark task" looks like, and provides a small
set of mock tools agents can call.

Why mock tools, not real ones?
The point of this workflow is to compare how well different models/
frameworks choose and use tools correctly - not to test real weather APIs
or real calendars. Mock tools return realistic-looking canned data, which
makes scoring deterministic and repeatable: the same task always has the
same correct answer, run after run, regardless of whether a real API is up
or rate-limited.

Each benchmark task lives in benchmarks/tool_use/*.yaml — see
benchmarks/tool_use/weather_and_calendar.yaml for a worked example.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import yaml


@dataclass
class ExpectedCall:
    """One tool call we expect a correctly-behaving agent to make."""
    tool: str
    args: Dict[str, Any] = field(default_factory=dict)
    required: bool = True


@dataclass
class ToolUseTask:
    """One benchmark task: a prompt, the tools available, and what a
    correct agent run should look like."""
    task_id: str
    prompt: str
    available_tools: List[str]
    expected_calls: List[ExpectedCall]


def load_task(yaml_path: str) -> ToolUseTask:
    """Load a single benchmark task from a YAML file."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    expected_calls = [
        ExpectedCall(
            tool=c["tool"],
            args=c.get("args", {}),
            required=c.get("required", True),
        )
        for c in data.get("expected_calls", [])
    ]

    return ToolUseTask(
        task_id=data["task_id"],
        prompt=data["prompt"],
        available_tools=data["available_tools"],
        expected_calls=expected_calls,
    )


# --- Mock tool implementations ---
#
# These simulate real-world tools with deterministic, canned responses.
# Add a new mock tool here whenever a new benchmark task needs one, and
# register it in TOOL_REGISTRY at the bottom of this file.

def get_weather(location: str, date: str = "today") -> dict:
    """Mock weather lookup — always returns rain for Boston, sunny elsewhere."""
    if "boston" in location.lower():
        return {"location": location, "date": date, "condition": "rain", "precipitation": True}
    return {"location": location, "date": date, "condition": "sunny", "precipitation": False}


def get_calendar_event(time: str) -> dict:
    """Mock calendar lookup — always returns one fake meeting."""
    return {"time": time, "title": "Team sync", "attendees": 4}


def update_calendar_event(new_time: str) -> dict:
    """Mock calendar update — always 'succeeds'."""
    return {"status": "updated", "new_time": new_time}


def currency_convert(amount: float, from_currency: str, to_currency: str) -> dict:
    """Mock currency conversion using a fixed, fake exchange rate."""
    fake_rate = 0.92 if to_currency.upper() == "EUR" else 1.0
    return {"amount": amount, "converted": round(amount * fake_rate, 2), "rate": fake_rate}


def calculator(expression: str) -> dict:
    """Mock calculator — evaluates simple arithmetic only."""
    try:
        # Restricted eval: digits and basic operators only, never arbitrary code.
        allowed = set("0123456789+-*/.() ")
        if not set(expression) <= allowed:
            raise ValueError("Expression contains disallowed characters")
        return {"expression": expression, "result": eval(expression)}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


TOOL_REGISTRY = {
    "get_weather": get_weather,
    "get_calendar_event": get_calendar_event,
    "update_calendar_event": update_calendar_event,
    "currency_convert": currency_convert,
    "calculator": calculator,
}
