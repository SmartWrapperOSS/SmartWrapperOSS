# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

# SmartWrapperOSS — Testing Guide

This README covers how to set up and test the codebase locally. For the
project's purpose, architecture, license, and disclaimers, see the
top-level README in the repo root (not included in this zip — this is
the code package only).

## What this project is

A model and workflow evaluator. You give it a task (summarize a
document, or run a tool-calling benchmark), it runs that task through
multiple orchestration frameworks (AutoGen-style, LangGraph-style) and
multiple LLMs, then scores and compares the results in one table.

It is **not** an agent product, and it doesn't integrate with any
specific vendor's desktop tools — it stays model- and framework-neutral
on purpose, so its scores are trustworthy regardless of which provider
you favor.

---

## 1. Setup

### 1.1 Create a virtual environment

```bash
cd smartwrappeross
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
```

A virtual environment keeps this project's dependencies isolated from
your system Python — recommended for any real use, not just testing.

### 1.2 Install dependencies

```bash
pip install -r requirements.txt
```

(If you intentionally skipped the virtual environment in 1.1 and are
installing into system Python instead, add `--break-system-packages` to
the command above.)

### 1.3 Create your config file

```bash
cp config/config.yaml.example config/config.yaml
```

Open `config/config.yaml` and fill in real API keys for whichever
providers you want to test against. You don't need all of them — only
add keys for the models you actually plan to run.

**Never commit `config/config.yaml`** — it's already in `.gitignore`,
since it holds real secrets.

### 1.4 (Summarization workflow only) GCS credentials

If you want to test the summarization workflow against a real document,
you'll need:
- A Google Cloud Storage bucket with a document uploaded to it (PDF,
  DOCX, TXT, or CSV)
- A service account JSON key with read access to that bucket, saved as
  `gcp-credentials.json` in the project root (also gitignored)

If you only want to test the **tool-use workflow**, you can skip this
entirely — it doesn't touch GCS at all.

---

## 2. Quick smoke test — no API keys needed

Before spending any real API money, you can confirm the codebase itself
is wired together correctly using Python's built-in mocking, with a
**fake** model that returns scripted responses instead of calling a real
API.

Create a file called `smoke_test.py` in the project root:

```python
import sys
from unittest.mock import patch
from core.model_router import ModelRouter, ModelResponse
from workflows.tool_use.runner_autogen import ToolUseAutoGenRunner
from workflows.tool_use.evaluator import ToolUseEvaluator
from workflows.tool_use.tasks import load_task
from core.types import EvalResult
from output.formatter import print_table, save_json

# Scripted fake responses simulating a model that completes the task correctly
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
task = load_task("benchmarks/tool_use/weather_and_calendar.yaml")

with patch.object(ModelRouter, "call", fake_call):
    result = ToolUseAutoGenRunner(router).run("fake-model", task)

print("Tool calls made:", [c["tool_name"] for c in result.extra["tool_calls"]])

evaluator = ToolUseEvaluator()
scored = evaluator.score_all([("autogen", "fake-model", result)], task)
results = [EvalResult(framework=fw, model_id=m, task_result=tr, score_result=sr)
           for fw, m, tr, sr in scored]

print_table(results)
print("Smoke test passed.")
```

Run it:

```bash
python smoke_test.py
```

**Expected output:** a comparison table with one row (`autogen` /
`fake-model`), all dimensions scoring 100, and `Smoke test passed.`
printed at the end. If this works, the core pipeline (runner → evaluator
→ formatter) is wired correctly — any issues after this point are
specific to a real provider's API, not the project's internal logic.

Delete `smoke_test.py` once you're done — it's a throwaway script, not
part of the shipped project.

---

## 3. Testing the Tool-Use workflow for real

Once your `config/config.yaml` has at least one real API key:

```bash
python main.py --workflow tool-use \
    --task benchmarks/tool_use/weather_and_calendar.yaml \
    --frameworks autogen langgraph \
    --models gpt-4o claude-3-5-sonnet
```

This will:
1. Load the benchmark task
2. Run it through both frameworks × both models (4 combinations)
3. Score each one mechanically against the task's `expected_calls`
4. Print a comparison table and save `results.json`

**Cheapest way to test with real money on the line:** use just one
framework and one cheap/fast model first:

```bash
python main.py --workflow tool-use \
    --task benchmarks/tool_use/weather_and_calendar.yaml \
    --frameworks autogen \
    --models gpt-4o
```

---

## 4. Testing the Summarization workflow for real

Requires GCS setup from step 1.4.

```bash
python main.py --workflow summarize \
    --file gs://your-bucket/your-document.pdf \
    --frameworks autogen langgraph \
    --models gpt-4o claude-3-5-sonnet
```

Note this workflow uses an LLM-as-judge for 3 of its 5 scoring
dimensions, so it makes more API calls per run than tool-use does (each
judged dimension is a separate call) — expect higher cost and latency
for the same number of (framework, model) combinations.

---

## 5. Things worth checking when you test

- **`results.json`** is written to the path set in `config.yaml` →
  `output.json_path` (default: `results.json` in the project root).
  Confirm it contains a `dimensions` object per result — if it's empty
  or missing fields, something failed silently upstream (check the
  terminal output for `failed: ...` lines).
- **Composite score sorting** — the table should always show the highest
  `composite_score` first (labeled "Best:" at the bottom of the table).
- **Anti-bias judge rotation** (summarization only) — if you set
  `judge_model` in config to the same model you're also evaluating,
  confirm in the output that a *different* model was actually used to
  judge it (check the `reason` text for plausibility — a judge scoring
  its own output is the bug this rule exists to prevent).
- **Tool-use efficiency scoring** — try editing a benchmark YAML to
  intentionally make a step impossible (e.g., remove a tool from
  `available_tools` that's required by `expected_calls`) and confirm the
  Task Completion score drops accordingly, rather than crashing.

---

## 6. Adding your own benchmark task (tool-use)

Copy `benchmarks/tool_use/weather_and_calendar.yaml`, change `task_id`,
`prompt`, `available_tools`, and `expected_calls`. If you need a tool
that doesn't exist yet, add a mock implementation to
`workflows/tool_use/tasks.py`'s `TOOL_REGISTRY`.

---

## 7. Known limitations to keep in mind while testing

- `MAX_TURNS = 6` in both tool-use runners caps how many tool-calling
  turns an agent gets before being cut off — a genuinely complex task
  with more than ~6 required steps will fail Task Completion even if the
  model "would have" finished, simply by running out of turns.
- The `calculator` mock tool only supports basic arithmetic
  (`+ - * / ( )`) — anything else will return an error result, which is
  expected behavior, not a bug.
- Cost figures are estimates based on the hardcoded rates in
  `core/model_router.py`'s `COST_PER_1K_TOKENS` — these will drift out
  of date as providers change pricing. Update them periodically.
