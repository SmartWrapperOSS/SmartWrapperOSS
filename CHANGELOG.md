# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-08-08

Initial public release.

### Added

- **Summarization workflow** — benchmark document summarization across
  orchestration frameworks and models. Documents are loaded from a
  configured GCS bucket, chunked per `config.yaml`. Scored on Quality,
  Coverage, and Conciseness (LLM-as-judge with anti-bias judge rotation),
  plus Latency and Cost.
- **Tool-Use workflow** — benchmark tool-calling agents against YAML-defined
  benchmark tasks. Scored on Tool Selection, Argument Correctness, Task
  Completion, and Efficiency (mechanical scoring against a known-correct
  task definition), plus Latency and Cost. Ships with one benchmark task
  (`weather_and_calendar.yaml`).
- **Framework runners** for both workflows in AutoGen-style and
  LangGraph-style orchestration (simplified, illustrative implementations —
  see README for scope notes).
- **Shared model router** supporting GPT-4o, Claude, Gemini, and any
  OpenAI-compatible or local endpoint.
- **Shared `Workflow` / `Evaluator` plugin interface** (`workflows/base.py`)
  so new workflow types can be added without touching the CLI, comparison
  table, or model router.
- **Multi-run statistics** — `--runs N` repeats each (framework, model)
  combination N times and reports mean ± standard deviation with 95%
  confidence intervals via `core/stats.py`, addressing single-run score
  noise.
- **Resumable, incremental runs** — raw per-run records are persisted to
  disk by `core/runstore.py`. Re-running the same command skips runs
  already completed, so an interrupted `--runs 10` resumes instead of
  restarting, and bumping `--runs 3` to `--runs 10` later only executes
  the additional 7.
- **`--max-cost`** — stops launching new runs once a session's cumulative
  API spend crosses a threshold; already-completed runs are kept and the
  same command resumes the rest later.
- **Concurrent execution** of (framework, model) combinations via a thread
  pool, with per-combination progress and running session cost printed as
  results land.
- **HTML dashboard output** (`output/dashboard.py`), in addition to the
  console comparison table and `results.json`, for both single-run and
  aggregated multi-run results.
- **Config system** (`config/config.yaml`) for API keys, model list, GCS
  bucket, per-workflow scoring weights, and optional prompt overrides for
  the summarization and tool-use runners.
- `TESTING.md` walkthrough, including a no-API-key smoke test.
- `CONTRIBUTING.md` and Developer Certificate of Origin (DCO) process for
  contributions — no CLA required.
- `ENHANCEMENTS.md` — a working list of scoped, self-contained contribution
  ideas for both existing workflows.

### Known limitations

- Framework runners represent AutoGen and LangGraph in simplified form and
  are not a substitute for either SDK in production.
- Summarization judge scores are model-generated assessments and can drift
  as underlying judge models are updated server-side by providers — see
  README Disclaimer.
- Tool-calling runners issue one tool call per turn (no parallel/batched
  tool calls yet — tracked in `ENHANCEMENTS.md`).
- Summarization workflow currently supports one document per run.

[Unreleased]: https://github.com/SmartWrapperOSS/SmartWrapperOSS/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/SmartWrapperOSS/SmartWrapperOSS/releases/tag/v0.1.0
