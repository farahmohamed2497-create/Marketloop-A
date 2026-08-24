"""Agent registry: tracks which agents exist and which tools each one
may call.

This is intentionally decoupled from MarketLoopMCPServer's existing
role-based filtering (SessionContext / active_role). That mechanism
stays as-is. This adds a second, orthogonal filter — per-agent tool
enablement — driven by the admin panel and stored in the same
database everything else already uses.

Design: opt-out. A tool is visible to an agent unless a row exists in
Agent_Tools for (agent_id, tool_name) with enabled=0. This means
newly-registered tools show up for every agent by default, and an
admin explicitly turns individual tools off (or back on) per agent.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .db import get_connection


class AgentRegistryError(RuntimeError):
    """Raised when an agent-registry operation is invalid."""


DEFAULT_AGENT_ID = "default"
PRODUCTION_AGENTS = {
    "rag-agent": "RAG Agent",
    "planning-agent": "Planning Agent",
    "state-graph-agent": "State Graph Agent",
}

SYSTEM_AGENTS = {
    "default",
    "test-agent",
}

def ensure_agent(
    agent_id: str,
    display_name: str | None = None,
    agent_type: str | None = None,
) -> None:
    """Create the agent row if it doesn't exist yet; always bump last_seen_at."""
    now = datetime.now(timezone.utc).isoformat()

    with get_connection() as connection:
        existing = connection.execute(
            "SELECT agent_id FROM Agents WHERE agent_id = ?",
            (agent_id,),
        ).fetchone()

        if existing is None:
            connection.execute(
                """
                INSERT INTO Agents (agent_id, display_name, agent_type, last_seen_at)
                VALUES (?, ?, ?, ?)
                """,
                (agent_id, display_name or agent_id, agent_type, now),
            )
        else:
            connection.execute(
                "UPDATE Agents SET last_seen_at = ? WHERE agent_id = ?",
                (now, agent_id),
            )

        connection.commit()


def list_agents() -> list[dict[str, Any]]:
    """Return every known agent plus how many tools are explicitly disabled."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT a.agent_id, a.display_name, a.description, a.agent_type,
                   a.created_at, a.last_seen_at,
                   COALESCE(SUM(CASE WHEN t.enabled = 0 THEN 1 ELSE 0 END), 0) AS disabled_tool_count
            FROM Agents AS a
            LEFT JOIN Agent_Tools AS t ON t.agent_id = a.agent_id
            GROUP BY a.agent_id
            ORDER BY a.display_name
            """
        ).fetchall()

        return [dict(row) for row in rows]


def get_disabled_tool_names(agent_id: str) -> set[str]:
    """Tool names explicitly turned off for this agent."""
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT tool_name FROM Agent_Tools WHERE agent_id = ? AND enabled = 0",
            (agent_id,),
        ).fetchall()

        return {row["tool_name"] for row in rows}


def get_agent_tool_state(
    agent_id: str,
    all_tool_names: list[str],
) -> list[dict[str, Any]]:
    """Full enabled/disabled state for every known tool, for this agent."""
    disabled = get_disabled_tool_names(agent_id)

    return [
        {"tool_name": name, "enabled": name not in disabled}
        for name in sorted(all_tool_names)
    ]


def filter_enabled(agent_id: str, tool_names: list[str]) -> list[str]:
    """Given a candidate tool-name list, drop the ones this agent has disabled."""
    disabled = get_disabled_tool_names(agent_id)
    return [name for name in tool_names if name not in disabled]


def set_tool_enabled(
    agent_id: str,
    tool_name: str,
    enabled: bool,
    updated_by: str | None = None,
) -> None:
    """Enable or disable a single tool for a single agent (runtime, no redeploy)."""
    if not tool_name:
        raise AgentRegistryError("tool_name is required")

    now = datetime.now(timezone.utc).isoformat()

    with get_connection() as connection:
        agent_exists = connection.execute(
            "SELECT 1 FROM Agents WHERE agent_id = ?",
            (agent_id,),
        ).fetchone()

        if agent_exists is None:
            raise AgentRegistryError(f"Unknown agent: {agent_id}")

        if enabled:
            # Opt-out model: "enabled" just means no disabling row survives.
            connection.execute(
                "DELETE FROM Agent_Tools WHERE agent_id = ? AND tool_name = ?",
                (agent_id, tool_name),
            )
        else:
            connection.execute(
                """
                INSERT INTO Agent_Tools (agent_id, tool_name, enabled, updated_at, updated_by)
                VALUES (?, ?, 0, ?, ?)
                ON CONFLICT(agent_id, tool_name)
                DO UPDATE SET enabled = 0, updated_at = excluded.updated_at, updated_by = excluded.updated_by
                """,
                (agent_id, tool_name, now, updated_by),
            )

        connection.commit()