"""Diagnostic: how does call_tool actually signal a disabled/unknown tool?

Does it raise a Python exception (McpError etc.), or return a normal
CallToolResult object with isError=True and no exception at all?

Run:
    py -m tests.diagnose_calltool_error_shape
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mcp_server import agent_registry
from mcp_server.config import get_database_path

AGENT_ID = "diag-calltool-agent"
TARGET_TOOL = "inventory_tool"


def _cleanup() -> None:
    conn = sqlite3.connect(get_database_path())
    conn.execute("DELETE FROM Agent_Tools WHERE agent_id = ?", (AGENT_ID,))
    conn.execute("DELETE FROM Agents WHERE agent_id = ?", (AGENT_ID,))
    conn.commit()
    conn.close()


async def run() -> None:
    agent_registry.ensure_agent(AGENT_ID, display_name=AGENT_ID)
    agent_registry.set_tool_enabled(AGENT_ID, TARGET_TOOL, enabled=False)

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-u", "-m", "mcp_server.server"],
        env={**os.environ, "PYTHONUNBUFFERED": "1", "MARKETLOOP_AGENT_ID": AGENT_ID},
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            try:
                result = await session.call_tool(TARGET_TOOL, {})
                print("No exception was raised. Here's the raw result object:")
                print("  type:", type(result))
                print("  isError:", getattr(result, "isError", "NO isError ATTRIBUTE"))
                print("  content:", result.content)
            except Exception as exc:
                print(f"An exception WAS raised: {type(exc).__name__}: {exc}")

    _cleanup()


if __name__ == "__main__":
    asyncio.run(run())