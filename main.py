# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

"""
main.py
SmartWrapperOSS CLI entry point.

Usage (summarization):
    python main.py --workflow summarize --file gs://bucket/doc.pdf \
                   --frameworks autogen langgraph \
                   --models gpt-4o claude-3-5-sonnet gemini-pro

Usage (tool-use):
    python main.py --workflow tool-use --task benchmarks/tool_use/weather_and_calendar.yaml \
                   --frameworks autogen langgraph \
                   --models gpt-4o claude-3-5-sonnet gemini-pro

HOW THIS FILE STAYS SIMPLE AS WORKFLOWS ARE ADDED:
This file never imports anything summarization- or tool-use-specific
directly into its logic. Instead, WORKFLOW_REGISTRY below maps a
--workflow name to a small "builder" function that knows how to set up
that workflow's runners, evaluator, and load its input. Adding a third
workflow means adding one registry entry — main()'s actual control flow
(ingest input -> run all combos -> evaluate -> print/save) never changes.
"""

import argparse
import concurrent.futures
import yaml

from core.model_router import ModelRouter
from output.formatter import print_table, save_json
from output.dashboard import save_html


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# --- Workflow registry -----------------------------------------------------
#
# Each entry tells main() how to build that workflow's pieces. To add a
# new workflow (e.g. "rag"), write a build_<name>() function with the same
# shape as the two below, then add one line here.

def build_summarize(config: dict, router: ModelRouter, args):
    from ingestion.document_loader import DocumentLoader
    from workflows.summarization.runner_autogen import SummarizationAutoGenRunner
    from workflows.summarization.runner_langgraph import SummarizationLangGraphRunner
    from workflows.summarization.evaluator import SummarizationEvaluator

    print("Ingesting document...")
    loader = DocumentLoader(credentials_path=config["gcs"].get("credentials_path"))
    chunks = loader.load(
        args.file,
        chunk_size=config["chunking"]["chunk_size"],
        overlap=config["chunking"]["chunk_overlap"],
    )
    print(f"  {len(chunks)} chunks extracted")

    task_input = chunks
    source_text_for_scoring = "\n\n".join(c.text for c in chunks[:5])  # first 5 chunks as reference

    # Optional prompt overrides from config/config.yaml's `prompts:` section.
    # If absent, each runner falls back to its own built-in default prompt —
    # see the `prompts:` block in config/config.yaml.example for the format.
    prompt_overrides = config.get("prompts", {}).get("summarize", {})
    summarize_kwargs = {}
    if "summarize_prompt" in prompt_overrides:
        summarize_kwargs["summarize_prompt"] = prompt_overrides["summarize_prompt"]
    if "review_prompt" in prompt_overrides:
        summarize_kwargs["review_prompt"] = prompt_overrides["review_prompt"]

    runners = {
        "autogen": SummarizationAutoGenRunner(router, **summarize_kwargs),
        "langgraph": SummarizationLangGraphRunner(router, **summarize_kwargs),
    }
    evaluator = SummarizationEvaluator(
        router=router,
        judge_model_id=config["evaluation"]["judge_model"],
        weights=config["evaluation"].get("summarization_weights"),
    )
    return runners, evaluator, task_input, source_text_for_scoring


def build_tool_use(config: dict, router: ModelRouter, args):
    from workflows.tool_use.runner_autogen import ToolUseAutoGenRunner
    from workflows.tool_use.runner_langgraph import ToolUseLangGraphRunner
    from workflows.tool_use.evaluator import ToolUseEvaluator
    from workflows.tool_use.tasks import load_task

    print(f"Loading benchmark task: {args.task}")
    task = load_task(args.task)
    print(f"  Task: {task.task_id}")

    task_input = task
    scoring_reference = task  # the evaluator scores against the task's expected_calls

    # Optional prompt override from config/config.yaml's `prompts:` section.
    prompt_overrides = config.get("prompts", {}).get("tool_use", {})
    tool_use_kwargs = {}
    if "agent_prompt" in prompt_overrides:
        tool_use_kwargs["agent_prompt"] = prompt_overrides["agent_prompt"]

    runners = {
        "autogen": ToolUseAutoGenRunner(router, **tool_use_kwargs),
        "langgraph": ToolUseLangGraphRunner(router, **tool_use_kwargs),
    }
    evaluator = ToolUseEvaluator(weights=config["evaluation"].get("tool_use_weights"))
    return runners, evaluator, task_input, scoring_reference


WORKFLOW_REGISTRY = {
    "summarize": build_summarize,
    "tool-use": build_tool_use,
}


# --- Main run loop -----------------------------------------------------
#
# This function has NO knowledge of summarization or tool-use specifics —
# it only talks to the generic Workflow/Evaluator interfaces. That's what
# keeps it short and unchanged as new workflows are added.

def _task_tag(workflow_name: str, args) -> str:
    """A short identifier for WHICH task was run, so repeated runs of
    different tasks are never aggregated together as if they were
    repeats of the same experiment (see core/runstore.py layout)."""
    import os
    if workflow_name == "summarize":
        return os.path.splitext(os.path.basename(args.file))[0]
    if workflow_name == "tool-use":
        return os.path.splitext(os.path.basename(args.task))[0]
    return "default"


def run(workflow_name: str, frameworks: list, model_ids: list, config: dict, args):
    print(f"\nSmartWrapperOSS — workflow: {workflow_name}")
    print(f"Frameworks: {', '.join(frameworks)}")
    print(f"Models:     {', '.join(model_ids)}")
    if args.runs > 1:
        print(f"Runs:       {args.runs} per combination (mean ± std dev, 95% CI)")
    print()

    router = ModelRouter(config["models"])
    build_fn = WORKFLOW_REGISTRY[workflow_name]
    runners, evaluator, task_input, scoring_reference = build_fn(config, router, args)

    from core.runstore import RunStore
    from core.stats import aggregate_runs
    store = RunStore(args.runs_dir, workflow_name, _task_tag(workflow_name, args))

    combos = [
        (fw, model_id)
        for fw in frameworks
        for model_id in model_ids
        if model_id in router.configs and fw in runners
    ]

    # Which (framework, model, run_index) still need executing? Runs
    # already persisted on disk are skipped — this makes --runs N both
    # RESUMABLE (a crash at run 4/5 never re-pays for runs 1-3) and
    # INCREMENTAL (--runs 3 today, --runs 10 next week just adds 7).
    pending = []
    for fw, model_id in combos:
        done = store.completed_indices(fw, model_id)
        resumed = len(done & set(range(args.runs)))
        if resumed:
            print(f"  resume: {fw} + {model_id}: {resumed}/{args.runs} runs already on disk")
        pending.extend((fw, model_id, i) for i in range(args.runs) if i not in done)

    completed = []  # (framework, model_id, run_index, TaskResult)
    spent = 0.0     # cumulative API cost of THIS session, for --max-cost
    if pending:
        print(f"\nRunning {len(pending)} executions "
              f"({len(combos)} combos x {args.runs} run(s), minus resumed)...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = {
                executor.submit(runners[fw].run, model_id, task_input): (fw, model_id, i)
                for fw, model_id, i in pending
            }
            for future in concurrent.futures.as_completed(futures):
                fw, model_id, i = futures[future]
                if future.cancelled():
                    continue
                try:
                    task_result = future.result()
                    completed.append((fw, model_id, i, task_result))
                    spent += task_result.cost_usd
                    print(f"  done: {fw} + {model_id} (run {i + 1})"
                          f"   session cost so far: ${spent:.4f}")
                except Exception as e:
                    print(f"  failed: {fw} + {model_id} (run {i + 1}): {e}"
                          f" — rerun the same command to retry just this run")
                if args.max_cost is not None and spent >= args.max_cost:
                    cancelled = sum(1 for f in futures if f.cancel())
                    if cancelled:
                        print(f"  --max-cost ${args.max_cost:.2f} reached: cancelled "
                              f"{cancelled} unstarted run(s). Completed runs are saved; "
                              f"rerun the same command to resume.")
    else:
        print("\nAll requested runs already on disk — skipping to aggregation.")

    if completed:
        print("\nEvaluating...")
        # score_all() preserves input order in both existing evaluators
        # (they iterate the input list), which lets us zip run indices
        # back onto the scored rows. The assert guards that contract.
        scored = evaluator.score_all(
            [(fw, m, tr) for fw, m, _, tr in completed], scoring_reference
        )
        for (fw, model_id, run_idx, _), (fw2, m2, tr, sr) in zip(completed, scored):
            assert (fw, model_id) == (fw2, m2), "Evaluator.score_all reordered results"
            store.save(fw, model_id, run_idx, {
                "workflow": workflow_name,
                "output": tr.output,
                "input_tokens": tr.input_tokens,
                "output_tokens": tr.output_tokens,
                "latency_ms": tr.latency_ms,
                "cost_usd": tr.cost_usd,
                "extra": tr.extra,
                "dimensions": sr.dimensions,
                "composite_score": sr.composite_score,
            })
        print(f"Raw run records saved to {store.dir}/ — publish these alongside "
              f"results so anyone can independently re-derive the numbers.")

    json_path = f"results_{workflow_name}.json"
    html_path = f"results_{workflow_name}.html"
    show_reasons = config["output"].get("show_reasons", True)

    records = store.load_all()
    max_n = max(
        (sum(1 for r in records
             if (r["framework"], r["model_id"]) == (fw, m)) for fw, m in combos),
        default=0,
    )

    if max_n > 1:
        # Multi-run mode: the headline output is the aggregate table.
        # Latency/cost dimension scores are batch-normalized per session,
        # so the aggregate reports RAW latency (median/p95) and cost
        # (mean) instead — see core/stats.py for the full rationale.
        aggs = aggregate_runs(records)
        from output.formatter import print_aggregate_table, save_aggregate_json
        print_aggregate_table(aggs)
        save_aggregate_json(aggs, f"results_{workflow_name}_aggregate.json")
        if config["output"].get("html_dashboard", True):
            from output.dashboard import save_aggregate_html
            save_aggregate_html(aggs, records,
                                f"results_{workflow_name}_aggregate.html",
                                workflow_name=workflow_name)
        return

    # Single-run mode: original behavior, unchanged.
    from core.types import EvalResult
    results = [
        EvalResult(framework=fw, model_id=model_id, task_result=tr, score_result=sr)
        for (fw, model_id, _, tr), (_, _, _, sr) in zip(completed, scored)
    ] if completed else []
    results.sort(key=lambda r: r.composite_score, reverse=True)

    print_table(results, show_reasons=show_reasons)
    save_json(results, json_path)
    if config["output"].get("html_dashboard", True):
        save_html(results, html_path, workflow_name=workflow_name)


def main():
    parser = argparse.ArgumentParser(
        description="SmartWrapperOSS: compare LLMs and orchestration frameworks on a task, with objective scoring."
    )
    parser.add_argument("--workflow", required=True, choices=list(WORKFLOW_REGISTRY.keys()),
                        help="Which task type to run.")
    parser.add_argument("--frameworks", nargs="+", default=["autogen", "langgraph"],
                        choices=["autogen", "langgraph"])
    parser.add_argument("--models", nargs="+", default=["gpt-4o", "claude-3-5-sonnet"])
    parser.add_argument("--config", default="config/config.yaml")

    # Multi-run / statistics flags
    parser.add_argument("--runs", type=int, default=1,
                        help="Repeat each (framework, model) combination N times and "
                             "report mean ± std dev with 95%% confidence intervals. "
                             "Raw runs are persisted to --runs-dir and resumable: "
                             "rerunning the same command only executes missing runs.")
    parser.add_argument("--runs-dir", default="runs",
                        help="Directory for raw per-run JSON records (default: runs/).")
    parser.add_argument("--max-cost", type=float, default=None,
                        help="Stop launching new runs once this session's cumulative "
                             "API cost (USD) exceeds this value. Completed runs are "
                             "kept; rerun to resume.")

    # Workflow-specific inputs — only one is required depending on --workflow
    parser.add_argument("--file", help="GCS URI, required for --workflow summarize")
    parser.add_argument("--task", help="Path to a benchmark task YAML, required for --workflow tool-use")

    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if args.workflow == "summarize" and not args.file:
        parser.error("--workflow summarize requires --file")
    if args.workflow == "tool-use" and not args.task:
        parser.error("--workflow tool-use requires --task")

    config = load_config(args.config)
    run(args.workflow, args.frameworks, args.models, config, args)


if __name__ == "__main__":
    main()
