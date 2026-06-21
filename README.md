# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

# SmartWrapperOSS

An open-source model and workflow evaluation framework.

Run the same task — document summarization, or a tool-calling agent
benchmark — through multiple orchestration frameworks (AutoGen-style,
LangGraph-style) and multiple LLMs, and compare quality, latency, and
cost side by side, in one command.

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
git clone https://github.com/yourname/smartwrappeross
cd smartwrappeross
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp config/config.yaml.example config/config.yaml   # add your API keys
```

See [TESTING.md](./TESTING.md) for a full setup walkthrough, including a
no-API-key smoke test.

### Run the Tool-Use evaluator

```bash
python main.py --workflow tool-use \
    --task benchmarks/tool_use/weather_and_calendar.yaml \
    --frameworks autogen langgraph \
    --models gpt-4o claude-sonnet-4-6
```

### Run the Summarization evaluator

```bash
python main.py --workflow summarize \
    --file gs://your-bucket/document.pdf \
    --frameworks autogen langgraph \
    --models gpt-4o claude-sonnet-4-6 gemini-3.5-flash
```

## Output

Summarization:
```
Framework    Model            Quality  Coverage  Concise  Latency   Cost     Score
────────────────────────────────────────────────────────────────────────────────────
LangGraph    claude-3-5-sonnet   91       88       79      1.8s    $0.018    87.4
AutoGen      claude-3-5-sonnet   88       85       82      2.1s    $0.019    84.7
LangGraph    gpt-4o              85       90       71      2.3s    $0.022    83.1
AutoGen      gpt-4o              83       87       74      2.6s    $0.024    81.2
LangGraph    gemini-pro          79       83       80      3.1s    $0.015    79.8
```

Tool-Use:
```
Framework    Model         Tool Selection  Arg Correctness  Completion  Efficiency  Latency  Cost    Score
─────────────────────────────────────────────────────────────────────────────────────────────────────────
LangGraph    claude-3-5-sonnet   100             100            100         100       1.6s   $0.004   100.0
AutoGen      claude-3-5-sonnet   100             100            100         100       1.9s   $0.005    98.7
```

Results from either workflow are also saved to `results.json` with full
outputs and scoring reasons.

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
        ┌─────────────────┴─────────────────┐
        │                                   │
  Summarization Workflow              Tool-Use Workflow
  (Document → chunks via GCS)         (YAML benchmark task)
        │                                   │
  ┌─────┴─────┐                       ┌─────┴─────┐
  AutoGen   LangGraph                 AutoGen   LangGraph
  runner     runner                   runner     runner
        │                                   │
        └─────────────────┬─────────────────┘
                          │
                  Model Router (shared)
              GPT-4o / Claude / Gemini / Llama
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
  Summarization Evaluator              Tool-Use Evaluator
  Quality · Coverage ·                 Tool Selection · Args ·
  Conciseness · Latency · Cost         Completion · Efficiency ·
  (LLM-as-judge, anti-bias rotation)   Latency · Cost (mechanical)
        │                                   │
        └─────────────────┬─────────────────┘
                          │
            Comparison Table + results.json
              (generic — renders whatever
               dimensions the workflow scored)
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

Copyright 2026 MY_NAME (SmartWrapperOSS)

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

- **The LLM judge avoids judging its own output, with some limits. The Quality, Coverage, and Conciseness scores in the summarization workflow are produced by a configured judge model (currently claude-sonnet-4-6). SummarizationEvaluator automatically substitutes a different model as judge whenever the configured judge is also the model under evaluation, to avoid a model rating its own output favorably. Two limits to be aware of: (1) this check only excludes the exact same model ID — it does not control for bias toward other models from the same provider or model family; and (2) if a judge call fails or returns unparseable output, that dimension silently falls back to a neutral score of 50 rather than failing the run, so an unusually "average" score on one dimension may reflect a judge error rather than a genuine middling result. We have not independently audited this benchmark for either form of residual bias.

- ** Published results reflect a small, fixed sample — not a statistically validated benchmark. Scores shown here come from a limited number of runs against a fixed set of benchmark tasks defined by the SmartWrapperOSS maintainer. They have not been repeated across multiple trials, multiple document types, or multiple prompt variations, and have not been independently reviewed. Small differences in score (for example, a few points) should not be read as a meaningful or reproducible difference in real-world performance. Treat published comparisons as illustrative of how the tool works, not as a vendor ranking you should act on without running your own evaluation against your own tasks.

- ** This compares capability tiers, not equivalent models. The models listed in any given table may differ substantially in size, training, and intended use case (for example, a smaller/cheaper model alongside a larger flagship model). This is not an apples-to-apples evaluation of "which model is best" in the abstract — it is a guide to help you judge what level of capability, quality, and cost your specific task actually requires. A lower score for a smaller model does not mean it is a worse model; it may simply be more than sufficient, and considerably cheaper, for your use case. Use these results to scope the right tier of model for your work, not to rank providers or models against each other in general.

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
  time; always verify against your provider's billing dashboard.

- **Data handling.**
  - Documents you upload (summarization workflow) are sent to **your
    own** Google Cloud Storage bucket and to whichever LLM APIs you
    configure.
  - Tool-use benchmark tasks run entirely against mock, local tool
    implementations — no external services are called by the tools
    themselves.
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
