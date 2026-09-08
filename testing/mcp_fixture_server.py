# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

"""
testing/mcp_fixture_server.py

A PINNED, deterministic MCP server for testing the tool-use workflow's
MCPToolProvider — it exposes the exact same mock tool functions that
MockToolProvider already wraps, but speaks real MCP (JSON-RPC over
stdio, real JSON Schema tool specs) instead of a plain Python dict.

Why this file exists, not a live MCP server:
Any benchmark task pointed at a real external MCP server breaks the
determinism the tool-use workflow depends on (same task, same correct
answer, run after run). This server never leaves the local machine and
never calls anything external — it's the same canned data as before,
just served over the real protocol. See ENHANCEMENTS.md / the
mcp_server.pinned convention in tasks.py for the rule this enforces.

Run standalone to sanity-check it starts:
    python testing/mcp_fixture_server.py
(it will sit waiting for a client on stdio — that's expected, Ctrl+C to stop)
"""

from mcp.server.fastmcp import FastMCP

from workflows.tool_use.tasks import (
    get_weather,
    get_calendar_event,
    update_calendar_event,
    currency_convert,
    calculator,
)

mcp = FastMCP("smartwrappeross-pinned-test-tools")

mcp.tool()(get_weather)
mcp.tool()(get_calendar_event)
mcp.tool()(update_calendar_event)
mcp.tool()(currency_convert)
mcp.tool()(calculator)

if __name__ == "__main__":
    mcp.run(transport="stdio")
