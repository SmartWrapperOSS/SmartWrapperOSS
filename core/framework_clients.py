# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

"""
core/framework_clients.py

Builds each REAL framework's own chat-model client from a config.yaml
model entry, so the framework does native tool calling (OpenAI `tools`,
Anthropic `tool_use`) through its own code path, not ours.

Used by workflows that run the actual LangGraph / AutoGen packages (see
workflows/guess_game/). Those packages are optional extras:

    pip install -r requirements-frameworks.txt

Supported providers: openai, anthropic, openai_compatible. (google needs
langchain-google-genai / an AutoGen Gemini client and isn't wired yet.)
"""

DEFAULT_MAX_TOKENS = 1000


def _model_name(model_id: str, config: dict) -> str:
    return config.get("api_model", model_id)


def _missing(pkg: str) -> ImportError:
    return ImportError(
        f"{pkg} is not installed. This workflow runs the real frameworks: "
        f"pip install -r requirements-frameworks.txt"
    )


def make_langchain_model(model_id: str, config: dict):
    """A LangChain chat model (used by the LangGraph runner)."""
    provider = config["provider"]
    max_tokens = config.get("max_tokens", DEFAULT_MAX_TOKENS)
    if provider in ("openai", "openai_compatible"):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise _missing("langchain-openai")
        kwargs = {"model": _model_name(model_id, config), "max_tokens": max_tokens,
                  "api_key": config.get("api_key", "not-needed")}
        if provider == "openai_compatible":
            kwargs["base_url"] = config["base_url"]
            kwargs["timeout"] = config.get("timeout", 120)
        return ChatOpenAI(**kwargs)
    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise _missing("langchain-anthropic")
        return ChatAnthropic(model=_model_name(model_id, config), api_key=config["api_key"],
                             max_tokens=max_tokens)
    raise ValueError(f"Provider '{provider}' isn't supported for framework runs yet "
                     f"(supported: openai, anthropic, openai_compatible).")


# AutoGen needs capability info for models it doesn't recognize by name.
# These workflows only need function calling.
_AUTOGEN_MODEL_INFO = {
    "vision": False,
    "function_calling": True,
    "json_output": False,
    "family": "unknown",
    "structured_output": False,
}


def make_autogen_client(model_id: str, config: dict):
    """An AutoGen ChatCompletionClient (used by the AutoGen runner)."""
    provider = config["provider"]
    max_tokens = config.get("max_tokens", DEFAULT_MAX_TOKENS)
    if provider in ("openai", "openai_compatible"):
        try:
            from autogen_ext.models.openai import OpenAIChatCompletionClient
        except ImportError:
            raise _missing("autogen-ext[openai]")
        kwargs = {"model": _model_name(model_id, config), "max_tokens": max_tokens,
                  "api_key": config.get("api_key", "not-needed"),
                  "model_info": _AUTOGEN_MODEL_INFO}
        if provider == "openai_compatible":
            kwargs["base_url"] = config["base_url"]
            kwargs["timeout"] = config.get("timeout", 120)
        return OpenAIChatCompletionClient(**kwargs)
    if provider == "anthropic":
        try:
            from autogen_ext.models.anthropic import AnthropicChatCompletionClient
        except ImportError:
            raise _missing("autogen-ext[anthropic]")
        return AnthropicChatCompletionClient(model=_model_name(model_id, config),
                                             api_key=config["api_key"], max_tokens=max_tokens,
                                             model_info=_AUTOGEN_MODEL_INFO)
    raise ValueError(f"Provider '{provider}' isn't supported for framework runs yet "
                     f"(supported: openai, anthropic, openai_compatible).")
