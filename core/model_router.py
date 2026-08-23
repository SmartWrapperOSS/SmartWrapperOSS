# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

"""
core/model_router.py

Sends a prompt to whichever LLM provider is configured, and returns a
standard response shape (text + token counts + latency + cost).

This file does NOT know or care which workflow is calling it. It has no
concept of "summarization" or "tool use" — it just sends prompts and
returns text. That's deliberate: it's the one piece of infrastructure
every workflow shares, so it should stay as simple and generic as possible.

Adding a genuinely NEW provider protocol (not just a new model) requires
adding one new `_call_x` method and one line in `_dispatch`. But most
open-weight models (Llama, Mistral, Qwen, DeepSeek, ...) don't need that:
any local runtime (Ollama, vLLM, LM Studio, llama.cpp server,
text-generation-webui) or hosted OSS-model API (Together, Groq, Fireworks,
OpenRouter) that speaks the OpenAI chat-completions wire format is already
covered by the `openai_compatible` provider below — adding one of those
models is purely a `config.yaml` edit, no code change required.
"""

import time
from dataclasses import dataclass


@dataclass
class ModelResponse:
    """Raw response from a single LLM call."""
    model_id: str
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float


# Published pricing per 1,000 tokens, in USD, for models with a known,
# stable list price. Update these as providers change their pricing — see
# the README disclaimer: these are estimates, always verify against your
# provider's billing dashboard.
#
# This table is only a fallback. Open-weight / self-hosted / third-party
# models have no single canonical price (a local Llama is free; the same
# weights served by a hosted API are not), so those should set
# `cost_per_1k_input` / `cost_per_1k_output` directly in config.yaml
# instead of being added here — see config/config.yaml.example.
COST_PER_1K_TOKENS = {
    "gpt-4o":            {"input": 0.0025,  "output": 0.010},
    "claude-sonnet-4-6": {"input": 0.003,  "output": 0.015},
    "gemini-3.5-flash":  {"input": 0.0015, "output": 0.009},
}


class ModelRouter:
    """
    Usage:
        router = ModelRouter(config["models"])
        response = router.call("gpt-4o", "Summarize this: ...")
    """

    def __init__(self, model_configs: list):
        # model_configs looks like:
        # [{"id": "gpt-4o", "provider": "openai", "api_key": "..."}, ...]
        self.configs = {m["id"]: m for m in model_configs}

    def call(self, model_id: str, prompt: str) -> ModelResponse:
        """Send `prompt` to `model_id` and return the response."""
        if model_id not in self.configs:
            raise ValueError(
                f"Model '{model_id}' is not configured. "
                f"Available models: {list(self.configs.keys())}"
            )
        config = self.configs[model_id]
        return self._dispatch(model_id, config, prompt)

    def _dispatch(self, model_id: str, config: dict, prompt: str) -> ModelResponse:
        provider = config["provider"]
        if provider == "openai":
            return self._call_openai(model_id, config, prompt)
        elif provider == "anthropic":
            return self._call_anthropic(model_id, config, prompt)
        elif provider == "google":
            return self._call_google(model_id, config, prompt)
        elif provider == "openai_compatible":
            return self._call_openai_compatible(model_id, config, prompt)
        elif provider == "mistral":
            raise ValueError(
                f"Model '{model_id}' uses provider 'mistral', which has been "
                f"replaced by the more general 'openai_compatible' provider "
                f"(it covers Llama, Mistral, Qwen, DeepSeek, etc. via any "
                f"local runtime or hosted OSS-model API — see "
                f"config/config.yaml.example). Update config.yaml: set "
                f"provider: openai_compatible and base_url to your "
                f"endpoint's OpenAI-compatible root, e.g. "
                f"'http://localhost:11434/v1' for Ollama."
            )
        else:
            raise ValueError(f"Unknown provider: '{provider}'")

    def _call_openai(self, model_id: str, config: dict, prompt: str) -> ModelResponse:
        import openai
        client = openai.OpenAI(api_key=config["api_key"])

        start = time.time()
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
        )
        latency_ms = (time.time() - start) * 1000

        usage = resp.usage
        cost = self._calc_cost(model_id, usage.prompt_tokens, usage.completion_tokens, config)

        return ModelResponse(
            model_id=model_id,
            text=resp.choices[0].message.content,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
        )

    def _call_anthropic(self, model_id: str, config: dict, prompt: str) -> ModelResponse:
        import anthropic
        client = anthropic.Anthropic(api_key=config["api_key"])

        start = time.time()
        resp = client.messages.create(
            model=model_id,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = (time.time() - start) * 1000

        cost = self._calc_cost(model_id, resp.usage.input_tokens, resp.usage.output_tokens, config)

        return ModelResponse(
            model_id=model_id,
            text=resp.content[0].text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
        )

    def _call_google(self, model_id: str, config: dict, prompt: str) -> ModelResponse:
        import google.generativeai as genai
        genai.configure(api_key=config["api_key"])
        model = genai.GenerativeModel(model_id)

        start = time.time()
        resp = model.generate_content(prompt)
        latency_ms = (time.time() - start) * 1000

        input_tokens = model.count_tokens(prompt).total_tokens
        output_tokens = model.count_tokens(resp.text).total_tokens
        cost = self._calc_cost(model_id, input_tokens, output_tokens, config)

        return ModelResponse(
            model_id=model_id,
            text=resp.text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
        )

    def _call_openai_compatible(self, model_id: str, config: dict, prompt: str) -> ModelResponse:
        """
        Calls any endpoint that implements the OpenAI chat-completions wire
        format. This single adapter covers local runtimes (Ollama's `/v1`
        endpoint, vLLM, LM Studio, llama.cpp server, text-generation-webui)
        AND hosted OSS-model APIs (Together, Groq, Fireworks, OpenRouter) —
        so Llama, Mistral, Qwen, DeepSeek, etc. are all just a config.yaml
        entry away, not a new adapter.
        """
        import openai

        base_url = config.get("base_url")
        if not base_url:
            raise ValueError(
                f"Model '{model_id}' has provider 'openai_compatible' but no "
                f"'base_url' in config.yaml. Point it at the endpoint's "
                f"OpenAI-compatible root, e.g. 'http://localhost:11434/v1' "
                f"(Ollama), 'http://localhost:8000/v1' (vLLM / LM Studio), "
                f"or 'https://api.together.xyz/v1' (Together)."
            )

        # Most local runtimes don't check the API key at all; hosted APIs
        # (Together, Groq, OpenRouter, ...) require a real one. The
        # placeholder keeps the OpenAI client happy either way.
        client = openai.OpenAI(
            api_key=config.get("api_key", "not-needed"),
            base_url=base_url,
            # Local models — especially larger ones running on modest
            # hardware — can be far slower to a first token than a cloud
            # API. Default higher than the SDK's own default so a slow
            # local model doesn't get cut off mid-generation; still
            # overridable per-model via `timeout` in config.yaml.
            timeout=config.get("timeout", 120),
        )

        # The friendly `id` used elsewhere in SmartWrapperOSS (CLI
        # --models, cost table, results.json) doesn't have to match what
        # the endpoint expects on the wire. Set `api_model` in
        # config.yaml when they differ, e.g. id: llama-3.3-70b,
        # api_model: "llama3.3:70b".
        api_model = config.get("api_model", model_id)

        # Some runtimes accept extra request-body fields beyond the OpenAI
        # spec (e.g. Ollama's `keep_alive`, controlling how long it keeps
        # the model loaded in memory between calls). `extra_params` in
        # config.yaml is passed straight through; unrecognized fields are
        # simply ignored by runtimes that don't support them.
        extra_params = config.get("extra_params", {})

        start = time.time()
        try:
            resp = client.chat.completions.create(
                model=api_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=config.get("max_tokens", 1000),
                extra_body=extra_params if extra_params else None,
            )
        except openai.APIConnectionError as e:
            # This is the failure mode people actually hit with local
            # models: the runtime just isn't up yet. Give a hint tailored
            # to the common case (Ollama on its default port) and a
            # generic one otherwise, instead of a raw connection traceback.
            if ":11434" in base_url:
                hint = (
                    f"Is Ollama running? Try `ollama serve`, and make sure "
                    f"the model is pulled: `ollama pull {api_model}`."
                )
            else:
                hint = f"Is the server at {base_url} running and reachable?"
            raise ConnectionError(
                f"Could not reach '{model_id}' at {base_url}. {hint}"
            ) from e
        except openai.APITimeoutError as e:
            raise TimeoutError(
                f"Call to '{model_id}' at {base_url} timed out after "
                f"{config.get('timeout', 120)}s. Local models on modest "
                f"hardware can be slow — try raising `timeout` in "
                f"config.yaml for this model."
            ) from e
        latency_ms = (time.time() - start) * 1000

        # Some local runtimes omit usage stats on certain configurations;
        # degrade to 0 rather than crashing an otherwise-successful run.
        usage = resp.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        cost = self._calc_cost(model_id, input_tokens, output_tokens, config)

        return ModelResponse(
            model_id=model_id,
            text=resp.choices[0].message.content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
        )

    def _calc_cost(self, model_id: str, input_tokens: int, output_tokens: int,
                    config: dict = None) -> float:
        # Config-supplied pricing always wins: the router can't know the
        # rates for every hosted endpoint or self-hosted deployment in
        # advance, so any model.yaml entry with explicit
        # cost_per_1k_input/output overrides the built-in table (and is
        # required for local models to correctly show $0, or for hosted
        # OSS-model APIs that DO charge to show a real cost).
        if config and ("cost_per_1k_input" in config or "cost_per_1k_output" in config):
            rates = {
                "input": config.get("cost_per_1k_input", 0.0),
                "output": config.get("cost_per_1k_output", 0.0),
            }
        else:
            rates = COST_PER_1K_TOKENS.get(model_id, {"input": 0.0, "output": 0.0})
        return (input_tokens / 1000 * rates["input"]) + (output_tokens / 1000 * rates["output"])
