"""E2E test for F1: admin REMOVES a tool -> agent can no longer CALL it live.

The mirror image of test_e2e_add_tool: starts with the tool enabled and
working, then proves that after an admin removes it, the same running
agent session is rejected when it tries to call that tool -- not just
that it disappears from list_tools.

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


async def run() -> None:
    print(
        f"--- E2E: admin REMOVES '{TARGET_TOOL}' "
        f"-> '{AGENT_ID}' can no longer call it live ---"
    )

    try:
        agent_registry.ensure_agent(
            AGENT_ID,
            display_name=AGENT_ID,
        )

        # Start with the tool explicitly enabled.
        agent_registry.set_tool_enabled(
            AGENT_ID,
            TARGET_TOOL,
            enabled=True,
        )

        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-u", "-m", "mcp_server.server"],
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "MARKETLOOP_AGENT_ID": AGENT_ID,
            },
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # -----------------------------------------------------
                # 1. Tool works before removal.
                # -----------------------------------------------------
                result = await session.call_tool(
                    TARGET_TOOL,
                    {},
                )

                if getattr(result, "isError", False):
                    text = "".join(
                        getattr(block, "text", "")
                        for block in result.content
                    )
                    raise AssertionError(
                        f"FAIL: '{TARGET_TOOL}' should work before removal, "
                        f"but server returned an error: {text!r}"
                    )

                text = "".join(
                    getattr(block, "text", "")
                    for block in result.content
                )

                print(
                    "  [1/2] Before removing: "
                    f"tool call succeeded -> OK ({text!r})"
                )

                # -----------------------------------------------------
                # 2. Simulate the exact admin action.
                # -----------------------------------------------------
                agent_registry.set_tool_enabled(
                    AGENT_ID,
                    TARGET_TOOL,
                    enabled=False,
                )

                # -----------------------------------------------------
                # 3. Same MCP session, no restart.
                # -----------------------------------------------------
                result_after_removal = await session.call_tool(
                    TARGET_TOOL,
                    {},
                )

                if not getattr(
                    result_after_removal,
                    "isError",
                    False,
                ):
                    text = "".join(
                        getattr(block, "text", "")
                        for block in result_after_removal.content
                    )

                    raise AssertionError(
                        f"FAIL: '{TARGET_TOOL}' is still callable "
                        f"after removal: {text!r}"
                    )

                error_text = "".join(
                    getattr(block, "text", "")
                    for block in result_after_removal.content
                )

                print(
                    "  [2/2] After removing "
                    "(same session, no restart): "
                    f"rejected -> OK ({error_text!r})"
                )

        print(
            "\nPASSED: admin 'remove tool' action stops "
            "a live agent from calling it."
        )

    finally:
        # Always remove test data, even when an assertion fails.
        _cleanup()

if __name__ == "__main__":
    asyncio.run(run())