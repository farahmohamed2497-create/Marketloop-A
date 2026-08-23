"""E2E test for F1: the agents list reflects real MCP server state.

Spins up two independent agent processes with distinct agent_ids,
then queries the SAME admin API endpoint the dashboard calls
(GET /api/agents) and checks both show up with accurate data --
proving the list isn't hardcoded or stale.

PREREQUISITE: the admin API must already be running, since this test
hits it exactly the way the dashboard does:
    py -m uvicorn admin_api.main:app --port 8001

Run from the repo root (in a second terminal):
    py -m tests.test_e2e_agents_list
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import urllib.error
import urllib.request

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mcp_server import agent_registry
from mcp_server.config import get_database_path

API_BASE = "http://localhost:8001"
AGENT_A = "e2e-list-test-agent-a"
AGENT_B = "e2e-list-test-agent-b"
DISABLED_TOOL_FOR_B = "order_tool"


def _cleanup() -> None:
    conn = sqlite3.connect(get_database_path())
    for agent_id in (AGENT_A, AGENT_B):
        conn.execute("DELETE FROM Agent_Tools WHERE agent_id = ?", (agent_id,))
        conn.execute("DELETE FROM Agents WHERE agent_id = ?", (agent_id,))
    conn.commit()
    conn.close()


def _fetch_agents() -> list[dict]:
    try:
        with urllib.request.urlopen(f"{API_BASE}/api/agents", timeout=5) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Couldn't reach the admin API at :8001. Start it first with:\n"
            "  py -m uvicorn admin_api.main:app --port 8001"
        ) from exc


async def _connect_agent(agent_id: str) -> None:
    """Connect once (enough to trigger ensure_agent) and disconnect."""
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-u", "-m", "mcp_server.server"],
        env={**os.environ, "PYTHONUNBUFFERED": "1", "MARKETLOOP_AGENT_ID": agent_id},
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await session.list_tools()


async def run() -> None:
    print("--- E2E: admin dashboard agents list reflects real MCP server state ---")

    await _connect_agent(AGENT_A)
    print(f"  [setup] '{AGENT_A}' connected with all tools available")

    await _connect_agent(AGENT_B)
    agent_registry.set_tool_enabled(AGENT_B, DISABLED_TOOL_FOR_B, enabled=False)
    print(f"  [setup] '{AGENT_B}' connected with '{DISABLED_TOOL_FOR_B}' removed")

    agents = _fetch_agents()
    by_id = {a["agent_id"]: a for a in agents}

    assert AGENT_A in by_id, f"FAIL: '{AGENT_A}' missing from /api/agents response"
    assert AGENT_B in by_id, f"FAIL: '{AGENT_B}' missing from /api/agents response"
    print(f"  [1/2] Both connected agents appear in the live list -> OK")

    assert by_id[AGENT_A]["disabled_tool_count"] == 0, (
        f"FAIL: '{AGENT_A}' should show 0 disabled tools, "
        f"got {by_id[AGENT_A]['disabled_tool_count']}"
    )
    assert by_id[AGENT_B]["disabled_tool_count"] == 1, (
        f"FAIL: '{AGENT_B}' should show 1 disabled tool, "
        f"got {by_id[AGENT_B]['disabled_tool_count']}"
    )
    print(f"  [2/2] Disabled-tool counts match real per-agent state -> OK")

    print("\nPASSED: the agents list is live data, not hardcoded or stale.")
    _cleanup()


if __name__ == "__main__":
    asyncio.run(run())