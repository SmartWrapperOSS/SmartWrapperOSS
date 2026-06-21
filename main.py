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

def run(workflow_name: str, frameworks: list, model_ids: list, config: dict, args):
    print(f"\nSmartWrapperOSS — workflow: {workflow_name}")
    print(f"Frameworks: {', '.join(frameworks)}")
    print(f"Models:     {', '.join(model_ids)}")
    print()

    router = ModelRouter(config["models"])
    build_fn = WORKFLOW_REGISTRY[workflow_name]
    runners, evaluator, task_input, scoring_reference = build_fn(config, router, args)

    print("\nRunning task across all (framework, model) combinations...")
    task_results = []  # list of (framework, model_id, TaskResult)
    combos = [
        (fw, model_id)
        for fw in frameworks
        for model_id in model_ids
        if model_id in router.configs
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(runners[fw].run, model_id, task_input): (fw, model_id)
            for fw, model_id in combos
            if fw in runners
        }
        for future in concurrent.futures.as_completed(futures):
            fw, model_id = futures[future]
            try:
                task_result = future.result()
                task_results.append((fw, model_id, task_result))
                print(f"  done: {fw} + {model_id}")
            except Exception as e:
                print(f"  failed: {fw} + {model_id}: {e}")

    print("\nEvaluating...")
    scored = evaluator.score_all(task_results, scoring_reference)

    from core.types import EvalResult
    results = [
        EvalResult(framework=fw, model_id=model_id, task_result=tr, score_result=sr)
        for fw, model_id, tr, sr in scored
    ]
    results.sort(key=lambda r: r.composite_score, reverse=True)

    show_reasons = config["output"].get("show_reasons", True)
    print_table(results, show_reasons=show_reasons)
    save_json(results, config["output"]["json_path"])

    # HTML dashboard is opt-out, not opt-in — it costs nothing extra to
    # generate (same data as results.json, just rendered) and most people
    # testing locally will want something easier to skim than raw JSON.
    if config["output"].get("html_dashboard", True):
        html_path = config["output"].get("html_path", "results.html")
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

    # Workflow-specific inputs — only one is required depending on --workflow
    parser.add_argument("--file", help="GCS URI, required for --workflow summarize")
    parser.add_argument("--task", help="Path to a benchmark task YAML, required for --workflow tool-use")

    args = parser.parse_args()

    if args.workflow == "summarize" and not args.file:
        parser.error("--workflow summarize requires --file")
    if args.workflow == "tool-use" and not args.task:
        parser.error("--workflow tool-use requires --task")

    config = load_config(args.config)
    run(args.workflow, args.frameworks, args.models, config, args)


if __name__ == "__main__":
    main()
