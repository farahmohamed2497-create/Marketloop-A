"""E2E test for F1: admin REMOVES a tool -> agent can no longer CALL it live.

See test_e2e_add_tool.py's docstring: a rejected call returns a normal
CallToolResult with isError=True, it does not raise a Python exception.

Run from the repo root:
    py -m tests.test_e2e_remove_tool
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

AGENT_ID = "e2e-remove-test-agent"
TARGET_TOOL = "inventory_tool"  # no-arg, no DB dependency -> safe to actually call


def _cleanup() -> None:
    conn = sqlite3.connect(get_database_path())
    conn.execute("DELETE FROM Agent_Tools WHERE agent_id = ?", (AGENT_ID,))
    conn.execute("DELETE FROM Agents WHERE agent_id = ?", (AGENT_ID,))
    conn.commit()
    conn.close()


def _content_text(result) -> str:
    return "".join(getattr(block, "text", "") for block in result.content)


async def run() -> None:
    print(f"--- E2E: admin REMOVES '{TARGET_TOOL}' -> '{AGENT_ID}' can no longer call it ---")

    agent_registry.ensure_agent(AGENT_ID, display_name=AGENT_ID)
    # Start in the AVAILABLE state.
    agent_registry.set_tool_enabled(AGENT_ID, TARGET_TOOL, enabled=True)

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-u", "-m", "mcp_server.server"],
        env={**os.environ, "PYTHONUNBUFFERED": "1", "MARKETLOOP_AGENT_ID": AGENT_ID},
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            before = await session.call_tool(TARGET_TOOL, {})
            assert not before.isError, (
                f"FAIL: expected the tool to work before removal. "
                f"Got isError={before.isError}, content={_content_text(before)!r}"
            )
            print(f"  [1/2] Before removing: call succeeded -> OK  ({_content_text(before)!r})")

            # This is exactly what admin_api/main.py's PUT endpoint does when
            # an admin clicks "Remove" in the dashboard.
            agent_registry.set_tool_enabled(AGENT_ID, TARGET_TOOL, enabled=False)

            after = await session.call_tool(TARGET_TOOL, {})
            assert after.isError, (
                f"FAIL: '{TARGET_TOOL}' should be rejected after being removed, "
                f"on the SAME running session, with no restart. "
                f"Got isError={after.isError}, content={_content_text(after)!r}"
            )
            print(f"  [2/2] After removing (same session, no restart): rejected -> OK  ({_content_text(after)!r})")

    print("\nPASSED: admin 'remove tool' action stops a live agent from calling it.")
    _cleanup()


if __name__ == "__main__":
    asyncio.run(run())