"""E2E test for F1: admin ADDS a tool -> agent can actually CALL it live.

This goes one step further than the list_tools toggle test: it proves
the tool is functionally callable after being added back, not just
present in the tool list.

Run from the repo root:
    py -m tests.test_e2e_add_tool
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

AGENT_ID = "e2e-add-test-agent"
TARGET_TOOL = "inventory_tool"  # no-arg, no DB dependency -> safe to actually call


def _cleanup() -> None:
    conn = sqlite3.connect(get_database_path())
    conn.execute("DELETE FROM Agent_Tools WHERE agent_id = ?", (AGENT_ID,))
    conn.execute("DELETE FROM Agents WHERE agent_id = ?", (AGENT_ID,))
    conn.commit()
    conn.close()


async def run() -> None:
    print(f"--- E2E: admin ADDS '{TARGET_TOOL}' -> '{AGENT_ID}' can call it live ---")

    agent_registry.ensure_agent(AGENT_ID, display_name=AGENT_ID)
    # Start in the REMOVED state, simulating a tool an admin took away earlier.
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
                await session.call_tool(TARGET_TOOL, {})
                raise AssertionError(
                    f"FAIL: '{TARGET_TOOL}' should NOT be callable while removed."
                )
            except AssertionError:
                raise
            except Exception as exc:
                print(f"  [1/2] Removed tool correctly rejected: {type(exc).__name__} -> OK")

            # This is exactly what admin_api/main.py's PUT endpoint does when
            # an admin clicks "Add" in the dashboard.
            agent_registry.set_tool_enabled(AGENT_ID, TARGET_TOOL, enabled=True)

            result = await session.call_tool(TARGET_TOOL, {})
            text = "".join(getattr(block, "text", "") for block in result.content)
            assert "inventory tool" in text, (
                f"FAIL: expected the real tool output, got: {text!r}"
            )
            print(f"  [2/2] After adding back: tool call succeeded -> OK  (result: {text!r})")

    print("\nPASSED: admin 'add tool' action makes a live agent able to call it.")
    _cleanup()


if __name__ == "__main__":
    asyncio.run(run())