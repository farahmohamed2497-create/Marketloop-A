"""Explicit bridge from planning DAG nodes to the existing MarketLoop MCP tools."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable
from typing import Any

from mcp_server.tools.inventory import update_inventory_quantity
from mcp_server.tools.reports import generate_sales_audit_report


READ_ONLY_TOOLS = frozenset({"generate_sales_audit_report"})
MUTATING_TOOLS = frozenset({"update_inventory_quantity"})


class MarketLoopMCPExecutor:
    """Run allow-listed MCP tools, requiring explicit approval for writes."""

    def __init__(
        self,
        *,
        allow_mutations: bool = False,
        tools: dict[str, Callable[..., Any]] | None = None,
    ) -> None:
        self.allow_mutations = allow_mutations
        self._tools = tools or {
            "generate_sales_audit_report": generate_sales_audit_report,
            "update_inventory_quantity": update_inventory_quantity,
        }

    def execute(self, tool_name: str, arguments: dict[str, object]) -> str:
        if tool_name not in self._tools:
            raise ValueError(f"Unknown MarketLoop MCP tool: {tool_name}")
        if tool_name in MUTATING_TOOLS and not self.allow_mutations:
            raise PermissionError(
                "Planning writes require allow_mutations=True after manager approval."
            )

        payload = dict(arguments)
        if tool_name == "generate_sales_audit_report":
            # The report tool has a strict two-field contract. Ignore
            # hallucinated optional fields emitted by a planner instead of
            # failing a read-only benchmark run.
            payload = {
                key: value
                for key, value in payload.items()
                if key in {"start_date", "end_date"}
            }

        result = self._tools[tool_name](payload)
        if inspect.isawaitable(result):
            result = asyncio.run(result)
        return result if isinstance(result, str) else json.dumps(result, sort_keys=True)

    def can_execute(self, tool_name: str) -> bool:
        """Return whether a DAG tool binding names a real allow-listed tool."""
        return tool_name in self._tools
