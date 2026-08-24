"""E2E test for F1: admin tool toggle -> live MCP server.

Unlike the earlier smoke tests (which called MarketLoopMCPServer directly
in-process), this spawns the server the same way a real agent does --
as a subprocess over stdio via mcp.client.stdio.stdio_client -- and talks
to it only through the MCP protocol (ClientSession.list_tools /
call_tool). This is the same path agent/client.py uses.

Run from the repo root:
    py tests/test_admin_tool_toggle_e2e.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mcp_server import agent_registry

AGENT_ID = "e2e-toggle-test-agent"
TARGET_TOOL = "generate_delay_apology"


async def run() -> None:
    print(f"--- E2E: admin toggle reaches a live '{AGENT_ID}' session ---")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-u", "-m", "mcp_server.server"],
        env={**os.environ, "PYTHONUNBUFFERED": "1", "MARKETLOOP_AGENT_ID": AGENT_ID},
    )
    # Make sure we start from a clean state for this agent.
    agent_registry.ensure_agent(AGENT_ID, display_name=AGENT_ID)
    agent_registry.set_tool_enabled(AGENT_ID, TARGET_TOOL, enabled=True)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            before = {t.name for t in (await session.list_tools()).tools}
            assert TARGET_TOOL in before, (
                f"FAIL: expected '{TARGET_TOOL}' to be visible before any admin action. "
                f"Got: {sorted(before)}"
            )
            print(f"  [1/3] Before disabling: '{TARGET_TOOL}' visible -> OK")

            # This is exactly what admin_api/main.py's PUT endpoint does when
            # an admin clicks "Remove" in the dashboard.
            agent_registry.set_tool_enabled(AGENT_ID, TARGET_TOOL, enabled=False)

            after_disable = {t.name for t in (await session.list_tools()).tools}
            assert TARGET_TOOL not in after_disable, (
                f"FAIL: '{TARGET_TOOL}' should have disappeared from the SAME running "
                f"session after being disabled, with no restart. Got: {sorted(after_disable)}"
            )
            print(f"  [2/3] After disabling (same session, no restart): '{TARGET_TOOL}' gone -> OK")

            # Same as clicking "Add" back in the dashboard.
            agent_registry.set_tool_enabled(AGENT_ID, TARGET_TOOL, enabled=True)

            after_enable = {t.name for t in (await session.list_tools()).tools}
            assert TARGET_TOOL in after_enable, (
                f"FAIL: '{TARGET_TOOL}' should have reappeared after re-enabling. "
                f"Got: {sorted(after_enable)}"
            )
            print(f"  [3/3] After re-enabling: '{TARGET_TOOL}' back -> OK")

    print("\nPASSED: admin panel toggle reaches a live MCP session with no restart.")


if __name__ == "__main__":
    asyncio.run(run())