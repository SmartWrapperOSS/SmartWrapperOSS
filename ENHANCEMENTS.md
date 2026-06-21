# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

# Enhancement ideas for contributors

This is a working list of concrete, scoped improvements to the two
existing workflows. None of these require changing `main.py`,
`output/formatter.py`, or `core/model_router.py` — they're all
self-contained within a workflow's own files, which makes them good
entry points for a first contribution. See `workflows/base.py` for the
interface contract these all build on.

If you want to add an entirely new workflow type (not listed here, since
that's a bigger undertaking) instead of enhancing an existing one, see
the "Adding a new workflow" section in the main [README](./README.md).

---

## Tool-Use Evaluator

### More benchmark tasks (good first issue)
Only one task ships today (`benchmarks/tool_use/weather_and_calendar.yaml`).
Useful additions, roughly in order of difficulty:

- **Distractor-tool tasks** — give the agent extra tools it should *not*
  call (e.g., a `calculator` task where `web_search` and
  `currency_convert` are also available but irrelevant), to test
  precision, not just recall, in tool selection.
- **Tool-variant disambiguation** — two similarly-named tools where only
  one is correct for the situation (e.g., `get_weather` vs.
  `get_extended_forecast`, where the right choice depends on how far out
  the date is). Tests whether the model is pattern-matching on tool
  names or actually reasoning about applicability.
- **Multi-step pipelines** — a task requiring 3+ sequential calls where
  each step's output feeds the next (e.g., search a flight → book it →
  email confirmation). Stresses the `MAX_TURNS` cap and state-passing.
- **Failure/recovery tasks** — a mock tool that returns an error or
  "no results" response, to test whether the agent retries sensibly or
  just gives up / hallucinates success. This requires extending the mock
  tools in `workflows/tool_use/tasks.py` to support a "fails on first
  call" mode.

### New scoring dimensions
The current evaluator ships with four mechanical dimensions
(Tool Selection, Argument Correctness, Task Completion, Efficiency).
Two were deliberately deferred — see the project's design discussion —
and are good intermediate contributions:

- **Order Correctness** — score whether calls happened in a valid
  sequence when the task defines a dependency (e.g., must check weather
  *before* deciding whether to reschedule). Requires extending
  `ExpectedCall` in `workflows/tool_use/tasks.py` with an explicit
  `depends_on` field, and adding the scoring logic to
  `workflows/tool_use/evaluator.py`.
- **Recovery Behavior** — needs an LLM-as-judge call (unlike the other
  four dimensions), since "did it adapt sensibly to a failure" is
  qualitative. Look at `workflows/summarization/evaluator.py`'s
  `_ask_judge` method for the existing judge-call pattern to reuse.

### Parallel tool-call support
Both current runners (`runner_autogen.py`, `runner_langgraph.py`) call
tools one at a time, in sequence — even when a task's tools have no
dependency on each other (e.g., checking weather in 5 different cities).
A framework that can correctly batch independent calls into a single
turn should score better on Efficiency and Latency than one that can't —
but the current runners don't give a model the option to request
multiple tool calls in one response. Adding that support (and updating
the `AGENT_PROMPT` to allow a `"call_tools"` plural action) would let
this kind of comparison actually happen.

### More mock tools
`TOOL_REGISTRY` in `workflows/tool_use/tasks.py` currently has 5 tools.
More variety (e.g., a `search_flights` / `book_flight` pair, a
`send_email` tool, a `search_restaurant_availability` tool) unlocks more
realistic, more interesting benchmark tasks without touching any other
file.

---

## Summarization Evaluator

### Configurable judge ensembles
Today, exactly one model judges each dimension (with anti-bias rotation
to avoid self-judging — see `_pick_judge` in
`workflows/summarization/evaluator.py`). A worthwhile extension: let
config specify *multiple* judges per dimension and average their scores,
which would reduce the chance that one judge's idiosyncrasies skew a
result. This only touches `evaluator.py` and `config.yaml.example`.

### Caching judge calls
Every (framework, model) combination calls the judge 3 times (once per
subjective dimension). If a user re-runs the same comparison to test a
config tweak unrelated to scoring, those judge calls re-run from
scratch. A simple cache (hash of source text + summary text + dimension
→ cached score) would cut cost and latency on repeated runs without
changing the scoring logic itself.

### Pinned judge model versions
Judge scores can drift over time as providers update models server-side
behind the same model ID (see the Disclaimer in the README). Allowing
`config.yaml` to specify a pinned model version string per provider,
where the provider's API supports it, would make scores more comparable
across runs taken weeks or months apart.

### Additional summarization dimensions
The current five (Quality, Coverage, Conciseness, Latency, Cost) cover
the basics. Possible additions, each requiring a new prompt in
`JUDGE_PROMPTS` and a new entry in `DEFAULT_WEIGHTS`:

- **Readability** — is the summary written at an appropriate reading
  level for its likely audience?
- **Structure** — does the summary use the source document's own
  structure (e.g., preserving section headers) where that's useful, or
  does it flatten everything into one block?
- **Citation grounding** — for summaries that reference specific facts
  or figures, does the summary make it possible to trace each claim back
  to a specific part of the source?

### Multi-document summarization
Currently one workflow run summarizes one document. Comparing how
frameworks/models handle summarizing *across* several related documents
(e.g., a quarter's worth of related filings) would need changes to
`ingestion/document_loader.py` (accept multiple URIs) and both
summarization runners (handle multiple sources in the prompt), but
nothing in `core/` or `output/`.

---

## How to propose something not on this list

Open an issue describing the idea before submitting a large PR — see
[CONTRIBUTING.md](./CONTRIBUTING.md). Smaller, self-contained PRs
(one new benchmark task, one new mock tool, one new scoring dimension)
are easier to review and merge quickly than large multi-part changes.
