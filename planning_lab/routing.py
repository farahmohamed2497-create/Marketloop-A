"""Routing policy for the separate MarketLoop Sales Audit planning agent."""

from __future__ import annotations

from enum import StrEnum


class PlanningMethod(StrEnum):
    MCP = "mcp"
    PLAN_AND_SOLVE = "plan_and_solve"
    TREE_OF_THOUGHTS = "tree_of_thoughts"
    REFLEXION = "reflexion"
    LATS = "lats"


def route_sales_audit_subtask(subtask: str) -> PlanningMethod:
    """Choose a method based on the subtask's actual decision shape."""
    normalized = subtask.lower()
    if any(token in normalized for token in ("retrieve", "audit data", "mcp report")):
        return PlanningMethod.MCP
    if any(token in normalized for token in ("compare", "alternative", "risk", "recommend")):
        return PlanningMethod.TREE_OF_THOUGHTS
    if any(token in normalized for token in ("retry", "correct", "approval", "validate")):
        return PlanningMethod.REFLEXION
    if any(token in normalized for token in ("commit", "write", "inventory adjustment")):
        return PlanningMethod.LATS
    return PlanningMethod.PLAN_AND_SOLVE