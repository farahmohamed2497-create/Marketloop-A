"""Grounded evaluator for MarketLoop sales-audit actions.

Unlike the reference toolkit's randomized Environment, this evaluator checks a
candidate action against the real sales-audit MCP result and the SQLite role
records before LATS or Reflexion accepts it.
"""

from __future__ import annotations

import json
from typing import Protocol

from mcp_server.db import get_connection

from .models import EnvironmentFeedback


class ReportExecutor(Protocol):
    def execute(self, tool_name: str, arguments: dict[str, object]) -> str: ...


class SalesAuditEnvironment:
    """Validate a proposed restock action against real MarketLoop evidence."""

    candidate_contract = """Return one JSON object only, with this schema:
{"action":"restock","product_id":<integer>,"quantity_change":<positive integer>,"user_id":<integer>}
The product must appear in the sales audit's low-stock list and user_id must be
an Inventory Manager, Manager, or Warehouse Admin in MarketLoop."""

    def __init__(self, executor: ReportExecutor, start_date: str, end_date: str) -> None:
        self.executor = executor
        self.start_date = start_date
        self.end_date = end_date

    def evaluate(self, state: str) -> EnvironmentFeedback:
        issues: list[str] = []
        try:
            candidate = json.loads(state)
        except json.JSONDecodeError:
            return EnvironmentFeedback(
                success=False,
                score=0.0,
                details=["Candidate must be a JSON sales-audit action."],
            )
        if not isinstance(candidate, dict):
            return EnvironmentFeedback(
                success=False,
                score=0.0,
                details=["Candidate must be a JSON object."],
            )

        if candidate.get("action") != "restock":
            issues.append("Only the grounded restock action is supported for this subtask.")
        product_id = candidate.get("product_id")
        quantity_change = candidate.get("quantity_change")
        user_id = candidate.get("user_id")
        if not isinstance(product_id, int) or product_id < 1:
            issues.append("product_id must be a positive integer.")
        if not isinstance(quantity_change, int) or quantity_change <= 0:
            issues.append("quantity_change must be a positive integer.")
        if not isinstance(user_id, int) or user_id < 1:
            issues.append("user_id must be a positive integer.")
        if issues:
            return EnvironmentFeedback(success=False, score=0.0, details=issues)

        report = json.loads(
            self.executor.execute(
                "generate_sales_audit_report",
                {"start_date": self.start_date, "end_date": self.end_date},
            )
        )
        low_stock_ids = {
            item["product_id"]
            for item in report["inventory"]["low_stock_items"]
        }
        if product_id not in low_stock_ids:
            issues.append(f"Product {product_id} is not low stock in the MCP audit report.")

        with get_connection() as connection:
            row = connection.execute(
                """
                SELECT r.role_name
                FROM Users AS u JOIN Roles AS r ON r.role_id = u.role_id
                WHERE u.user_id = ?
                """,
                (user_id,),
            ).fetchone()
        allowed_roles = {"inventory manager", "manager", "warehouse admin"}
        if row is None or (row["role_name"] or "").strip().lower() not in allowed_roles:
            issues.append("user_id is not authorized to approve an inventory adjustment.")

        if issues:
            return EnvironmentFeedback(success=False, score=0.35, details=issues)
        return EnvironmentFeedback(
            success=True,
            score=1.0,
            details=["Restock action matches low-stock evidence and an authorized role."],
        )