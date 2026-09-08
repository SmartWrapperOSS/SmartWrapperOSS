# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

# SmartWrapperOSS

An open-source model and workflow evaluation framework.

Run the same task — document summarization, or a tool-calling agent
benchmark — through multiple orchestration frameworks (AutoGen-style,
LangGraph-style) and multiple LLMs, and compare quality, latency, and
cost side by side, in one command. Repeat runs with `--runs N` and get
mean ± std dev with 95% confidence intervals instead of single-sample
numbers.

## What this is

A neutral benchmarking tool. You give it a task, it runs that task
multiple ways, and it tells you — with real scores, not vibes — which
combination of framework and model actually performed best for that
task.

## What this is **not**

To keep expectations accurate, and to keep the project's scope from
drifting:

- **Not an agent product.** SmartWrapperOSS does not run autonomous
  workflows on your behalf, manage your files, or act as a personal or
  team assistant. It runs a defined task once, scores the result, and
  stops.
- **Not tied to any single AI vendor.** It is intentionally
  model-agnostic and has no special integration with any one provider's
  desktop or platform tools. This is what makes its comparisons
  trustworthy — it has no reason to favor one model over another.
- **Not a production orchestration framework.** AutoGen and LangGraph
  here are represented in simplified, illustrative form to make a fair,
  readable comparison — this project is not a substitute for either
  framework's actual SDK in a production system.
- **Not a source of ground truth.** Its LLM-as-judge scores (used in the
  summarization workflow) are model-generated assessments, not objective
  measurements — see the Disclaimer section below.

## Quickstart

```bash
git clone https://github.com/SmartWrapperOSS/SmartWrapperOSS
cd SmartWrapperOSS
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp config/config.yaml.example config/config.yaml   # add your API keys
```

See [TESTING.md](./TESTING.md) for a full setup walkthrough, including a
no-API-key smoke test.

## Installation

For local development, or to contribute, use the Quickstart above.

SmartWrapperOSS is not yet published to PyPI, but it can still be
installed with pip in either of these ways:

**From a local clone** (recommended if you also want the example config
and benchmark task files on disk):

```bash
git clone https://github.com/SmartWrapperOSS/SmartWrapperOSS
cd SmartWrapperOSS
pip install .
```

**Directly from GitHub, without cloning first** — useful for adding
SmartWrapperOSS as a dependency of another project:

```bash
pip install git+https://github.com/SmartWrapperOSS/SmartWrapperOSS.git@v0.1.0
```

The `@v0.1.0` pins to a specific release tag so the installed version
won't change under you later; drop it (or swap in a different tag) to
track `main` instead. This same line can also go directly in another
project's `requirements.txt`:

```
smartwrappeross @ git+https://github.com/SmartWrapperOSS/SmartWrapperOSS.git@v0.1.0
```

Either install method gives you the `core`, `workflows`, `ingestion`,
and `output` packages plus a `smartwrappeross` command on your PATH.
**Note:** `config/` and `benchmarks/` ship as data, not as part of the
installed Python package. Installing from a local clone (first option)
means those files are right there next to your checkout; installing
directly from GitHub (second option) does not include them, since pip
discards its internal clone after building the package. Either way, pass
`--config` / `--task` with paths to your own copies of
`config/config.yaml` and the benchmark task YAML if they're not in your
working directory already. A future release may remove this
restriction; see [ENHANCEMENTS.md](ENHANCEMENTS.md).

### Run the Tool-Use evaluator

```bash
python main.py --workflow tool-use \
    --task benchmarks/tool_use/weather_and_calendar.yaml \
    --frameworks autogen langgraph \
    --models gpt-4o claude-sonnet-4-6
```

Tasks can also source their tools from a real (but pinned, local) MCP
server instead of the built-in mocks — see
`benchmarks/tool_use/weather_and_calendar_mcp.yaml` for an example, and
`workflows/tool_use/tool_provider.py` for how a task opts in via an
`mcp_server:` block.

### Run the Summarization evaluator

```bash
python main.py --workflow summarize \
    --file gs://your-bucket/document.pdf \
    --frameworks autogen langgraph \
    --models gpt-4o claude-sonnet-4-6 gemini-3.5-flash
```


### Run with statistics (recommended for any published comparison)

The `--runs`, `--runs-dir`, and `--max-cost` flags work identically for
every workflow — they belong to the shared pipeline, not to any one task
type.

```bash
# Tool-use, 5 runs per (framework, model) combination
python main.py --workflow tool-use \
    --task benchmarks/tool_use/weather_and_calendar.yaml \
    --frameworks autogen langgraph \
    --models gpt-4o claude-sonnet-4-6 \
    --runs 5 --max-cost 2.50

# Summarization, same flags
python main.py --workflow summarize \
    --file gs://your-bucket/document.pdf \
    --frameworks autogen langgraph \
    --models gpt-4o claude-sonnet-4-6 gemini-3.5-flash \
    --runs 5 --max-cost 5.00
```

Note for summarization: multi-run variance is especially informative
there, because its Quality/Coverage/Conciseness scores come from an
LLM-as-judge and are inherently noisier run-to-run than the tool-use
workflow's mostly mechanical scoring. Raw runs are stored per task under
`runs/summarize/<document-name>/`, so repeated runs of *different*
documents are never mixed into one aggregate. Also budget accordingly:
summarization runs typically cost more per run than tool-use benchmarks
(hence the higher `--max-cost` in the example).

- **`--runs N`** repeats every (framework, model) combination N times and
  reports **mean ± std dev** per dimension and a **95% confidence
  interval** (Student's t, appropriate for small N) on the composite
  score, instead of a single-sample number.
- **Resumable.** Every run is persisted to `runs/<workflow>/<task>/` as
  one JSON file the moment it's scored. If run 4 of 5 dies on a rate
  limit, rerunning the same command executes only the missing run — you
  never pay for the same run twice.
- **Incremental.** `--runs 3` today and `--runs 10` next week just adds
  the 7 new runs; the aggregate grows to n=10 automatically.
- **`--max-cost X`** stops launching new runs once the session's
  cumulative API cost exceeds X dollars. Completed runs are kept.
- **Verifiable.** The raw per-run records (full outputs, token counts,
  latency, cost, scores) are plain JSON. Publishing the `runs/`
  directory alongside a results table lets anyone independently
  re-derive — or dispute — the numbers.

With `--runs 5` the comparison table becomes (real output, tool-use
benchmark, top two rows):

```
Aggregated over repeated runs (n=5); score cells are mean ±std dev; 95% CI on composite.
Framework    Model               Tool Selection   Arg Correctness   Task Completion   Efficiency   Lat med/p95   Cost mean    Composite
langgraph    gpt-4o                 100.0 ±0.0       100.0 ±0.0        100.0 ±0.0      53.3 ±7.5    4.1s/4.4s     $0.0063     91.4 ±1.2
    composite 95% CI: [89.9, 93.0]   range: [90.4, 93.5]   n=5
autogen      gpt-4o                 100.0 ±0.0        85.0 ±22.4       100.0 ±0.0      50.0 ±0.0    4.2s/4.4s     $0.0061     87.4 ±5.7
    composite 95% CI: [80.4, 94.5]   range: [78.4, 91.4]   n=5

Best (by mean composite): langgraph + gpt-4o (91.4 ±1.2, n=5)
Note: 95% CIs of the top two overlap — this ranking is NOT statistically
distinguishable at n=5. Add runs to tighten.
```

That last line is deliberate: when the data can't support a ranking, the
tool says so instead of implying a winner. A one-point lead at n=5 is
noise more often than signal, and pretending otherwise is exactly the
kind of benchmark theater this project exists to avoid.

Two statistical notes:

1. **Latency and cost aggregate on raw values** (median/p95 seconds,
   mean dollars), never on their 0–100 scores — those scores are
   normalized within a single comparison (see the note under Output
   below) and are mathematically invalid to average across runs.
2. Latency uses **median and p95** rather than mean because API latency
   is long-tailed; one slow call would otherwise skew the whole cell.

Aggregate results are also saved to
`results_<workflow>_aggregate.json` and, unless disabled via
`output.html_dashboard`, to `results_<workflow>_aggregate.html` — a
self-contained dashboard whose headline is an **error-bar chart**
(dot = mean composite, whiskers = 95% CI), with expandable per-combination
rows showing per-dimension stats and a per-run breakdown. No server
needed; it opens by double-clicking, same as the single-run dashboard.

## Output

The numbers below are **real, published results**: mean ± sample std dev
over **n=5 runs** per (framework, model) combination, produced with
`--runs 5`. The raw per-run records that generated them are committed in
[`runs/`](./runs) — every number in these tables can be independently
re-derived from those files.

Workflow also generates an HTML dashboard, not just results.json

**Tool-Use Workflow** (`benchmarks/tool_use/weather_and_calendar.yaml`, n=5)

| Framework | Model | Tool Selection | Argument Correctness | Task Completion | Efficiency | Latency med/p95 | Cost mean | Composite |
|-----------|-------|----------------|----------------------|-----------------|------------|-----------------|-----------|-----------|
| langgraph | gpt-4o | 100.0 ±0.0 | 100.0 ±0.0 | 100.0 ±0.0 | 53.3 ±7.5 | 4.1s / 4.4s | $0.0063 | **91.4 ±1.2** |
| autogen | gpt-4o | 100.0 ±0.0 | 85.0 ±22.4 | 100.0 ±0.0 | 50.0 ±0.0 | 4.2s / 4.4s | $0.0061 | 87.4 ±5.7 |
| autogen | claude-sonnet-4-6 | 100.0 ±0.0 | 100.0 ±0.0 | 100.0 ±0.0 | 66.7 ±0.0 | 10.2s / 11.0s | $0.0103 | 83.9 ±0.8 |
| langgraph | claude-sonnet-4-6 | 86.7 ±29.8 | 100.0 ±0.0 | 86.7 ±29.8 | 73.4 ±14.9 | 10.4s / 12.4s | $0.0087 | 80.0 ±7.6 |

What the error bars reveal that a single run cannot:

- **autogen + gpt-4o, Argument Correctness 85.0 ±22.4** — the arguments
  are perfect in most runs and badly wrong occasionally. A single-run
  benchmark would have reported either "100" or a failure depending
  purely on which run it happened to catch.
- **langgraph + claude-sonnet-4-6, 86.7 ±29.8 on Tool Selection and Task
  Completion** — four clean runs and one failed one. Intermittent
  failure is exactly the behavior repetition exists to surface, and it
  is invisible at n=1.
- The top two composites (91.4 ±1.2 vs 87.4 ±5.7) have **overlapping 95%
  CIs**, and the tool says so rather than declaring a winner.

**Summarization Workflow** (n=5)

| Framework | Model | Quality | Coverage | Conciseness | Latency med/p95 | Cost mean | Composite |
|-----------|-------|---------|----------|-------------|-----------------|-----------|-----------|
| langgraph | gpt-4o | 96.2 ±1.1 | 95.0 ±0.0 | 64.0 ±4.5 | 4.0s / 9.1s | $0.0077 | **90.5 ±2.9** |
| langgraph | gemini-3.5-flash | 97.0 ±1.2 | 97.2 ±0.5 | 62.0 ±7.1 | 13.9s / 14.0s | $0.0092 | 82.2 ±11.2 |
| autogen | gpt-4o | 91.4 ±6.3 | 94.4 ±1.3 | 46.0 ±5.5 | 7.5s / 7.8s | $0.0121 | 81.6 ±3.0 |
| autogen | gemini-3.5-flash | 97.2 ±0.5 | 97.0 ±1.2 | 54.0 ±4.5 | 15.0s / 16.4s | $0.0099 | 81.2 ±1.6 |
| langgraph | claude-sonnet-4-6 | 95.0 ±0.0 | 97.0 ±2.7 | 85.0 ±0.0 | 12.4s / 14.3s | $0.0187 | 81.0 ±1.0 |
| autogen | claude-sonnet-4-6 | 97.0 ±2.7 | 95.0 ±0.0 | 86.0 ±2.2 | 15.6s / 17.3s | $0.0224 | 76.7 ±2.1 |

Two things worth noticing here, because they illustrate how to read any
benchmark of this kind:

- **The composite winner is not everyone's winner.**
  claude-sonnet-4-6 ranks near the bottom on composite — driven largely
  by latency and cost — yet wins Conciseness by ~20+ points with almost
  zero variance (85.0 ±0.0, 86.0 ±2.2). If concise output is the thing
  your task actually needs, the "best" row in this table is not your
  best choice. This is precisely why we say these results scope the
  right capability tier for *your* task rather than rank models in the
  abstract.
- **A mean without its spread is misleading.**
  langgraph + gemini-3.5-flash ranks second by mean composite
  (82.2), but at ±11.2 its 95% CI spans [68.4, 96.1] — nearly the whole
  field. Its rank is real; its *stability* at that rank is not
  established. The four combos below it at ±3.0 or tighter are far more
  predictable.

> **About Latency and Cost in these tables:** the aggregate tables show
> **raw values** — median / p95 seconds and mean dollars per run.
> Internally, each comparison batch also computes normalized 0–100
> Latency/Cost *scores* that feed the composite; those scores rank the
> rows in one batch against each other (100 = cheapest/fastest shown,
> not "free") and are therefore never averaged across runs. Single-run
> output (`--runs 1`) displays those normalized scores with the raw
> value in parentheses.

Results are saved to `results_<workflow>.json` / `.html` for single
runs, and to `results_<workflow>_aggregate.json` /
`results_<workflow>_aggregate.html` (error-bar chart + per-run
breakdown) for multi-run mode, with raw per-run records in `runs/`.

Live, interactive versions of the published dashboards — error-bar
charts, per-run breakdowns, and the earlier single-run dashboards they
supersede — are at the
[results index](https://smartwrappeross.github.io/SmartWrapperOSS/results/).

## Architecture

Every workflow plugs into the same shared pipeline through two small
interfaces (`Workflow` and `Evaluator` — see `workflows/base.py`), so
adding a new task type never requires touching the CLI, comparison
table, or model router.

```
                          main.py (CLI)
                               │
                   picks a workflow by name
                               │
            ┌──────────────────┴──────────────────┐
            │                                      │
   Summarization Workflow                 Tool-Use Workflow
   (Document → chunks via GCS)            (YAML benchmark task)
            │                                      │
     ┌──────┴──────┐                         ┌──────┴──────┐
   AutoGen      LangGraph                  AutoGen      LangGraph
   runner        runner                    runner        runner
            │                                      │
            │                             ┌────────┴────────┐
            │                        Mock Tools      MCP Server
            │                       (TOOL_REGISTRY)  (real protocol,
            │                                          pinned & local)
            │                                      │
            └──────────────────┬──────────────────┘
                               │
                      Model Router (shared)
                  GPT-4o / Claude / Gemini / Llama
                               │
            ┌──────────────────┴──────────────────┐
            │                                      │
   Summarization Evaluator                 Tool-Use Evaluator
   Quality · Coverage ·                    Tool Selection · Args ·
   Conciseness · Latency · Cost            Completion · Efficiency ·
   (LLM-as-judge, anti-bias rotation)      Latency · Cost (mechanical)
            │                                      │
            └──────────────────┬──────────────────┘
                               │
                    Run Store (runs/*.json)
                 every run persisted raw — this is
                what makes --runs N resumable and the
                    published stats auditable
                               │
       Comparison Table + results/dashboards (.json / .html,
                  plus *_aggregate.json / *_aggregate.html)
             (generic — renders whatever dimensions the
              workflow scored; multi-run mode adds mean,
                 std dev, and 95% confidence intervals)
```

## Configuration

Edit `config/config.yaml` to set API keys, model list, GCS bucket
(summarization only), and per-workflow scoring weights.

## Adding a new workflow

SmartWrapperOSS is built so a third workflow (e.g. RAG/Q&A) can be added
without modifying the CLI, formatter, or model router — see the
docstring at the top of `workflows/base.py` for the exact steps, and
`workflows/tool_use/` for a complete worked example of a second workflow
added this way.

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md). Good first issues are labeled
in GitHub.

This project uses a **Developer Certificate of Origin (DCO)**, not a CLA —
contributing just requires adding `-s` to your commit (`git commit -s`).
See [DCO.md](./DCO.md) for details on why, and how.

## Project model: open core + hosted service

The code in this repository — ingestion, orchestration, the model router,
and the evaluation engine — is, and will remain, fully open source under
Apache 2.0. There is no plan to relicense this core to a more restrictive
license in the future.

Separately, a hosted/managed version of SmartWrapperOSS may be offered as a
paid commercial product (e.g., infrastructure, billing, a polished UI, and
enterprise features on top of the open core) — built and operated for
profit, not solely to cover costs. This is a common model among open
source projects: the core stays open and free, while a separate commercial
offering funds the business. That hosted layer, if and when it exists, is
maintained as separate, closed-source code and is not part of this
repository. Contributions to this repository only ever go into the open
core, under the same Apache 2.0 terms everyone else uses — they are never
moved into a closed-source product.

## License

Copyright 2026 Aditi Jain (SmartWrapperOSS)

Licensed under the Apache License, Version 2.0.
See the [LICENSE](./LICENSE) file in the root of this repository for full terms.

## Disclaimer

SmartWrapperOSS is an orchestration and evaluation tool that connects to
third-party model providers (e.g., OpenAI, Anthropic, Google, and any
OpenAI-compatible or local endpoint you configure). SmartWrapperOSS does not
host or operate these models itself, and is provided "AS IS" without warranty
of any kind — see [LICENSE](./LICENSE) for full terms.

- **Outputs may contain errors.** LLM-generated summaries and agent
  responses may contain inaccuracies, omissions, or incorrect tool
  usage. Validate outputs before relying on them for any decision.

- **Evaluation scores are not ground truth.** The Coverage, Conciseness,
  and Quality dimensions (summarization workflow) are produced by an
  LLM-as-judge pattern — model-generated assessments, not objective
  measurements, that can vary run-to-run as underlying judge models are
  updated by their providers. The Tool-Use workflow's dimensions are
  mostly mechanical (compared against a known-correct task definition),
  but still depend on how representative the benchmark task is of your
  real use case. Use all scores as a directional signal to support human
  decision-making, not as a substitute for it.

- **The LLM judge avoids judging its own output, with some limits.** The
  Quality, Coverage, and Conciseness scores in the summarization
  workflow are produced by a configured judge model (currently
  `claude-sonnet-4-6`). `SummarizationEvaluator` automatically
  substitutes a different model as judge whenever the configured judge
  is also the model under evaluation, to avoid a model rating its own
  output favorably. Two limits to be aware of: (1) this check only
  excludes the exact same model ID — it does not control for bias
  toward other models from the same provider or model family; and (2)
  if a judge call fails or returns unparseable output, that dimension
  silently falls back to a neutral score of 50 rather than failing the
  run, so an unusually "average" score on one dimension may reflect a
  judge error rather than a genuine middling result. We have not
  independently audited this benchmark for either form of residual
  bias.

- **Read the error bars — and be suspicious of results without them.**
  The tool supports repeated trials via `--runs N`, reporting mean ±
  std dev and 95% confidence intervals, and it explicitly flags when
  the top-ranked combinations are not statistically distinguishable at
  the current sample size. Raw per-run records are written to `runs/`
  so any published aggregate can be independently re-derived. That
  said: the example tables in this README report mean ± std dev over
  five runs of a single fixed benchmark task each; results still
  reflect a fixed set of benchmark tasks defined by the maintainer, not
  multiple document types or prompt variations; and confidence
  intervals quantify run-to-run sampling noise only — they do not
  correct for judge bias, task selection, or how representative the
  benchmark is of your workload. Treat any comparison (ours or anyone
  else's) that shows single-run point scores with no variance as
  illustrative at best, and run your own evaluation against your own
  tasks before acting on a ranking.

- **This compares capability tiers, not equivalent models.** The models
  listed in any given table may differ substantially in size, training,
  and intended use case (for example, a smaller/cheaper model alongside
  a larger flagship model). This is not an apples-to-apples evaluation
  of "which model is best" in the abstract — it is a guide to help you
  judge what level of capability, quality, and cost your specific task
  actually requires. A lower score for a smaller model does not mean it
  is a worse model; it may simply be more than sufficient, and
  considerably cheaper, for your use case. Use these results to scope
  the right tier of model for your work, not to rank providers or
  models against each other in general.

- **You are responsible for your own API usage and costs.** Running this
  tool calls third-party LLM APIs (OpenAI, Anthropic, Google, or any
  endpoint you configure). You must hold a valid API key/account with each
  provider and comply with their respective terms of service and usage
  policies, including
  [OpenAI's Usage Policies](https://openai.com/policies/usage-policies),
  [Anthropic's Usage Policy](https://www.anthropic.com/legal/aup), and
  [Google's Generative AI Prohibited Use Policy](https://policies.google.com/terms/generative-ai/use-policy).
  SmartWrapperOSS does not modify, waive, or substitute for those terms, and
  you are responsible for all costs incurred on your own accounts.

- **Cost figures are estimates, not guarantees.** The cost dimension is
  calculated from publicly published provider pricing combined with token
  counts returned in each response. Provider pricing can change at any
  time; always verify against your provider's billing dashboard. Pricing
  in this repository's example tables and default cost table was last
  verified against each provider's official documentation on 2026-06-21
  — if you're reading this later, treat published example costs as
  illustrative of the comparison, not as current pricing.

- **No affiliation with model providers.** Neither I nor SmartWrapperOSS
  are affiliated with, endorsed by, or employed by OpenAI, Anthropic, or
  Google. Model names (GPT-4o, Claude, Gemini), trademarks, and logos
  referenced in this project belong to their respective owners and are
  used only to identify which models are being benchmarked. This is an
  independent, third-party tool built on top of these providers' public
  APIs — nothing here represents an official integration, partnership,
  or statement from any of these companies.

- **Data handling.**
  - Documents you upload (summarization workflow) are sent to **your
    own** Google Cloud Storage bucket and to whichever LLM APIs you
    configure.
  - Tool-use benchmark tasks run against either built-in mock tools or a
    local MCP server you configure — both execute entirely on your own
    machine. A task's `mcp_server` block may only point at a pinned,
    local, deterministic server (enforced at load time); it may never
    point at a live external endpoint, so no external services are
    called by the tools themselves either way.
  - No data is sent to or stored by the SmartWrapperOSS maintainers.
  - You are solely responsible for ensuring your use complies with
    applicable data protection laws and obligations (e.g., GDPR, CCPA,
    HIPAA, confidentiality agreements, or data residency requirements)
    relevant to the documents you process. Do not process personally
    identifiable information (PII) or protected health information (PHI)
    through third-party LLM APIs without first verifying those providers'
    relevant compliance certifications.

- **No SLA.** This is a community project with no uptime, support, or
  reliability guarantees.
