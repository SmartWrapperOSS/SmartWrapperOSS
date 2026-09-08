# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

"""
workflows/tool_use/tool_provider.py

Where tool descriptions and tool execution actually come from, for the
tool-use workflow. This is the seam that lets ToolUseAutoGenRunner and
ToolUseLangGraphRunner run against either the built-in mock tools (the
default, fully deterministic) or a real MCP server (real protocol, real
JSON Schema), without either runner or the ToolUseEvaluator needing to
know which one is in play — both providers return the same normalized
shapes, so nothing downstream changes.

Two implementations:
    MockToolProvider - wraps TOOL_REGISTRY (tasks.py). Default behavior,
                        unchanged from before this file existed.
    MCPToolProvider  - talks to a real MCP server over stdio.

Why MCPToolProvider is more than a thin wrapper:
The MCP SDK is async-only. Every other part of this codebase (ModelRouter,
both runners) is synchronous. Rather than making the whole project async,
MCPToolProvider runs its own event loop on a background thread for the
lifetime of one task run, and exposes plain synchronous list_tools()/
call_tool() methods — the same shape MockToolProvider exposes. From the
runner's point of view, calling a tool never changes.
"""

import asyncio
import json
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from workflows.tool_use.tasks import TOOL_REGISTRY


@dataclass
class ToolSpec:
    """What a tool provider knows about one tool, before it's ever called."""
    name: str
    description: str
    input_schema: Dict[str, Any]


class ToolProvider(ABC):
    @abstractmethod
    def list_tools(self) -> List[ToolSpec]:
        raise NotImplementedError

    @abstractmethod
    def call_tool(self, name: str, args: dict) -> dict:
        """
        Must always return a plain dict, never raise — {"error": "..."}
        on failure, same as the mock tools already do. This is what lets
        ToolUseEvaluator score a result without knowing which provider
        produced it.
        """
        raise NotImplementedError

    def __enter__(self):
        """
        Default no-op context manager so callers can write
        `with build_tool_provider(task) as provider:` uniformly, whether
        they got a MockToolProvider (nothing to set up) or an
        MCPToolProvider (which overrides this to spawn its subprocess).
        """
        return self

    def __exit__(self, *exc_info):
        return False


class MockToolProvider(ToolProvider):
    """
    Wraps TOOL_REGISTRY. This is the default provider — no existing
    benchmark YAML has to add an mcp_server block to keep working.

    list_tools() derives a lightweight schema from each function's own
    signature (via inspect.signature), rather than returning {}. This
    isn't cosmetic: a docstring alone often doesn't name its arguments
    (see update_calendar_event's docstring, which never mentions
    `new_time`) — without real argument names, a model has to guess,
    frequently guesses wrong, and burns an extra tool call correcting
    itself. That's a real efficiency (and cost) hit on every mock-based
    run, independent of whether MCP is involved at all — see the
    Efficiency finding from the mock-vs-MCP comparison for a concrete
    before/after. Deriving from the signature closes that gap for free,
    and keeps the mock-vs-MCP comparison honest: the interesting
    variable is protocol realism, not "did we forget to name a param."
    """

    def __init__(self, tool_names: List[str]):
        self.tool_names = tool_names

    def list_tools(self) -> List[ToolSpec]:
        specs = []
        for name in self.tool_names:
            fn = TOOL_REGISTRY.get(name)
            doc = fn.__doc__.strip() if fn and fn.__doc__ else "(no description)"
            specs.append(ToolSpec(name=name, description=doc,
                                   input_schema=self._schema_from_signature(fn) if fn else {}))
        return specs

    def _schema_from_signature(self, fn) -> Dict[str, Any]:
        import inspect
        sig = inspect.signature(fn)
        properties = {}
        required = []
        type_map = {str: "string", int: "integer", float: "number", bool: "boolean"}
        for pname, param in sig.parameters.items():
            py_type = param.annotation if param.annotation is not inspect.Parameter.empty else str
            properties[pname] = {"type": type_map.get(py_type, "string")}
            if param.default is inspect.Parameter.empty:
                required.append(pname)
        return {"type": "object", "properties": properties, "required": required}

    def call_tool(self, name: str, args: dict) -> dict:
        fn = TOOL_REGISTRY.get(name)
        if fn is None:
            return {"error": f"Unknown tool: {name}"}
        try:
            return fn(**args)
        except TypeError as e:
            return {"error": f"Bad arguments: {e}"}


class MCPToolProvider(ToolProvider):
    """
    Talks to a real MCP server, spawned as a subprocess over stdio.

    A fresh instance is created PER run() CALL (see the runner changes),
    not shared across concurrent runs — main.py's ThreadPoolExecutor
    calls .run() concurrently across frameworks and across --runs N
    repeats, so each call gets its own subprocess and its own session.

    Use as a context manager around exactly one runner call:

        with MCPToolProvider("python", ["testing/mcp_fixture_server.py"]) as provider:
            ... use provider.list_tools() / provider.call_tool() ...
    """

    def __init__(self, command: str, args: Optional[List[str]] = None, env: Optional[dict] = None,
                 cwd: Optional[str] = None):
        self.command = command
        self.args = args or []
        # Defaults to the current working directory — main.py is always
        # invoked from the project root (see TESTING.md), which is what
        # lets a fixture server like testing/mcp_fixture_server.py import
        # `workflows.tool_use.tasks` without needing an absolute path
        # baked into every benchmark YAML.
        self.cwd = cwd or os.getcwd()
        # `cwd` alone doesn't put the project root on the spawned
        # process's sys.path (Python adds the *script's* directory, not
        # the working directory) — so a fixture server that imports
        # project modules like `workflows.tool_use.tasks` needs
        # PYTHONPATH set explicitly. Inherit the parent's environment
        # unless the caller passed their own, so things like PATH still
        # work.
        if env is None:
            env = dict(os.environ)
        env.setdefault("PYTHONPATH", self.cwd)
        self.env = env
        self._loop = None
        self._thread = None
        self._session = None
        self._ready = None      # threading.Event — safe to signal cross-thread
        self._stop_event = None  # asyncio.Event — must be signaled via call_soon_threadsafe
        self._run_future = None
        self._startup_error = None

    def __enter__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._ready = threading.Event()
        self._stop_event = asyncio.Event()

        # Both connect and disconnect happen inside this ONE coroutine,
        # which runs as a single asyncio task for the lifetime of the
        # provider. This matters: anyio (which the MCP SDK's stdio
        # transport uses under the hood) requires a cancel scope to be
        # exited from the same task it was entered in — running connect
        # and disconnect as two separate run_coroutine_threadsafe calls
        # (two different tasks) raises "Attempted to exit cancel scope
        # in a different task than it was entered in" on teardown.
        self._run_future = asyncio.run_coroutine_threadsafe(self._run(), self._loop)

        if not self._ready.wait(timeout=30):
            raise RuntimeError(f"MCP server ({self.command} {self.args}) did not start in time")
        if self._startup_error:
            raise self._startup_error
        return self

    def __exit__(self, *exc_info):
        self._loop.call_soon_threadsafe(self._stop_event.set)
        try:
            self._run_future.result(timeout=10)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
        return False

    async def _run(self):
        """Owns the whole connection lifetime: connect, signal ready,
        wait to be told to stop, then disconnect — all as one task."""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        try:
            params = StdioServerParameters(command=self.command, args=self.args, env=self.env, cwd=self.cwd)
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._ready.set()
                    await self._stop_event.wait()
        except Exception as e:
            self._startup_error = e
            self._ready.set()  # unblock __enter__ even on failure

    def list_tools(self) -> List[ToolSpec]:
        return asyncio.run_coroutine_threadsafe(self._list_tools(), self._loop).result(timeout=15)

    async def _list_tools(self) -> List[ToolSpec]:
        result = await self._session.list_tools()
        return [
            ToolSpec(name=t.name, description=t.description or "", input_schema=t.inputSchema or {})
            for t in result.tools
        ]

    def call_tool(self, name: str, args: dict) -> dict:
        return asyncio.run_coroutine_threadsafe(self._call_tool(name, args), self._loop).result(timeout=30)

    async def _call_tool(self, name: str, args: dict) -> dict:
        try:
            result = await self._session.call_tool(name, args)
        except Exception as e:
            return {"error": str(e)}

        if getattr(result, "isError", False):
            text = "".join(getattr(c, "text", "") for c in result.content)
            return {"error": text or "MCP tool call failed"}

        # Prefer structured content when the server provides it (typed
        # results); fall back to concatenated text content otherwise.
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            return structured

        text = "".join(getattr(c, "text", "") for c in result.content)
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return {"result": text}


def build_tool_provider(task) -> ToolProvider:
    """
    Shared helper both runners call at the top of run(): decide which
    provider this task needs. A task with no mcp_server block behaves
    exactly as before this file existed.
    """
    if getattr(task, "mcp_server", None):
        # `pinned` is a safety flag checked by load_task() (tasks.py) —
        # it's not a connection parameter, so it doesn't get passed
        # through to the subprocess spawn.
        server_config = {k: v for k, v in task.mcp_server.items() if k != "pinned"}
        return MCPToolProvider(**server_config)
    return MockToolProvider(task.available_tools)
