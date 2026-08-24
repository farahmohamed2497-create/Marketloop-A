"""Discover the full set of registered tool names, independent of any
single agent's live session.

Used by the admin API so it can show "all known tools" per agent
(including ones the agent currently has disabled) without needing a
running MCP client connection.
"""

from __future__ import annotations

from .server import MarketLoopMCPServer


def discover_all_tool_names() -> list[str]:
    """Import + scan tools/resources/prompts and return every tool's name."""
    probe = MarketLoopMCPServer(name="_admin_probe")
    probe.register_modules()

    return sorted(
        getattr(tool, "name", tool.__name__) for tool in probe._tools
    )