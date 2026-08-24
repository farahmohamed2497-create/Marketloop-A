"""Diagnostic: is a pre-disabled tool excluded from list_tools() at all?

This isolates whether the bug is in the list_tools path or specific
to call_tool's dispatch loop.

Run:
    py -m tests.diagnose_predisable
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

AGENT_ID = "diag-predisable-agent"
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

    # Confirm the DB actually shows it disabled, from this process.
    disabled_now = agent_registry.get_disabled_tool_names(AGENT_ID)
    print(f"DB says disabled for '{AGENT_ID}' right before connecting: {disabled_now}")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-u", "-m", "mcp_server.server"],
        env={**os.environ, "PYTHONUNBUFFERED": "1", "MARKETLOOP_AGENT_ID": AGENT_ID},
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = {t.name for t in (await session.list_tools()).tools}
            print(f"list_tools() from the live session shows: {sorted(tools)}")
            print(f"Is '{TARGET_TOOL}' present? {TARGET_TOOL in tools}")

    _cleanup()


if __name__ == "__main__":
    asyncio.run(run())