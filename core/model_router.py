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

Adding a new model provider only requires adding one new `_call_x` method
and one line in `_dispatch` — nothing else in the project needs to change.
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


# Published pricing per 1,000 tokens, in USD. Update these as providers
# change their pricing — see the README disclaimer: these are estimates,
# always verify against your provider's billing dashboard.
COST_PER_1K_TOKENS = {
    "gpt-4o":            {"input": 0.005,  "output": 0.015},
    "claude-3-5-sonnet": {"input": 0.003,  "output": 0.015},
    "gemini-pro":        {"input": 0.0005, "output": 0.0015},
    "llama3":            {"input": 0.0,    "output": 0.0},  # local model, no API cost
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
        elif provider == "ollama":
            return self._call_ollama(model_id, config, prompt)
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
        cost = self._calc_cost(model_id, usage.prompt_tokens, usage.completion_tokens)

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

        cost = self._calc_cost(model_id, resp.usage.input_tokens, resp.usage.output_tokens)

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
        cost = self._calc_cost(model_id, input_tokens, output_tokens)

        return ModelResponse(
            model_id=model_id,
            text=resp.text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
        )

    def _call_ollama(self, model_id: str, config: dict, prompt: str) -> ModelResponse:
        import requests
        base_url = config.get("base_url", "http://localhost:11434")

        start = time.time()
        resp = requests.post(
            f"{base_url}/api/generate",
            json={"model": model_id, "prompt": prompt, "stream": False},
        )
        latency_ms = (time.time() - start) * 1000
        data = resp.json()

        return ModelResponse(
            model_id=model_id,
            text=data["response"],
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            latency_ms=latency_ms,
            cost_usd=0.0,  # local models have no API cost
        )

    def _calc_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        rates = COST_PER_1K_TOKENS.get(model_id, {"input": 0.0, "output": 0.0})
        return (input_tokens / 1000 * rates["input"]) + (output_tokens / 1000 * rates["output"])
