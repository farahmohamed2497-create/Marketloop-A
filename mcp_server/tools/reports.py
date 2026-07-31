"""Asynchronous long-running report tools with MCP progress tracking."""

from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from ..db import get_connection

LOW_STOCK_THRESHOLD = 20

PROGRESS_STAGES: tuple[tuple[float, str], ...] = (
    (0.00, "Initializing report generation..."),
    (0.25, "Aggregating total sales revenue..."),
    (0.50, "Calculating return ratios and refund metrics..."),
    (0.75, "Checking inventory levels and system audit logs..."),
    (1.00, "Report complete."),
)

PROGRESS_DELAY_SECONDS = 0.1


class GenerateSalesAuditReportInput(BaseModel):
    """Strict input schema for generating a sales audit report."""

    start_date: str = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Report period start date (YYYY-MM-DD).",
    )
    end_date: str = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Report period end date (YYYY-MM-DD).",
    )

    model_config = {
        "extra": "forbid",
    }


class ToolValidationError(ValueError):
    """Raised when the input or business rules are invalid."""


async def _send_progress(context: Any, progress: float, message: str) -> None:
    """Send an intermediate progress notification when the client supports it."""
    if context is None:
        return
    session = getattr(context, "session", None)
    progress_token = getattr(context, "progress_token", None)
    if session is None or progress_token is None or not hasattr(session, "send_progress_notification"):
        return
    await session.send_progress_notification(
        progress_token=progress_token,
        progress=progress,
        total=1.0,
        message=message,
    )


async def _aggregate_sales(connection: Any, start_date: str, end_date: str) -> dict[str, Any]:
    """Aggregate order and line-item statistics over the reporting window."""
    totals = connection.execute(
        """
        SELECT COUNT(*) AS order_count, COALESCE(SUM(total_amount), 0) AS total_revenue
        FROM Orders
        WHERE order_date BETWEEN ? AND ?
        """,
        (start_date, end_date),
    ).fetchone()

    by_status = connection.execute(
        """
        SELECT status, COUNT(*) AS cnt
        FROM Orders
        WHERE order_date BETWEEN ? AND ?
        GROUP BY status
        """,
        (start_date, end_date),
    ).fetchall()

    items = connection.execute(
        """
        SELECT COALESCE(SUM(oi.quantity), 0) AS units_sold,
               COALESCE(SUM(oi.quantity * oi.price), 0) AS items_value
        FROM Order_Items AS oi
        JOIN Orders AS o ON oi.order_id = o.order_id
        WHERE o.order_date BETWEEN ? AND ?
        """,
        (start_date, end_date),
    ).fetchone()

    order_count = int(totals["order_count"])
    total_revenue = float(totals["total_revenue"])
    return {
        "total_orders": order_count,
        "total_revenue": round(total_revenue, 2),
        "units_sold": int(items["units_sold"]),
        "items_value": round(float(items["items_value"]), 2),
        "average_order_value": round(total_revenue / order_count, 2) if order_count else 0.0,
        "orders_by_status": {row["status"]: int(row["cnt"]) for row in by_status},
    }


async def _aggregate_returns(connection: Any, start_date: str, end_date: str) -> dict[str, Any]:
    """Calculate return request and refund metrics over the reporting window."""
    totals = connection.execute(
        """
        SELECT COUNT(*) AS total_returns,
               COALESCE(SUM(CASE WHEN status = 'Approved' THEN 1 ELSE 0 END), 0) AS approved,
               COALESCE(SUM(CASE WHEN status = 'Rejected' THEN 1 ELSE 0 END), 0) AS rejected,
               COALESCE(SUM(CASE WHEN status = 'Pending' THEN 1 ELSE 0 END), 0) AS pending
        FROM Return_Requests
        WHERE request_date BETWEEN ? AND ?
        """,
        (start_date, end_date),
    ).fetchone()

    return {
        "total_returns": int(totals["total_returns"]),
        "approved_returns": int(totals["approved"]),
        "rejected_returns": int(totals["rejected"]),
        "pending_returns": int(totals["pending"]),
    }


async def _aggregate_inventory_and_audit(connection: Any, start_date: str, end_date: str) -> dict[str, Any]:
    """Check inventory levels and audit log activity over the reporting window."""
    low_stock = connection.execute(
        """
        SELECT p.product_id, p.product_name, i.quantity
        FROM Inventory AS i
        JOIN Products AS p ON i.product_id = p.product_id
        WHERE i.quantity <= ?
        ORDER BY i.quantity ASC
        """,
        (LOW_STOCK_THRESHOLD,),
    ).fetchall()

    audit = connection.execute(
        """
        SELECT COUNT(*) AS audit_events
        FROM Audit_Log
        WHERE date(action_time) BETWEEN ? AND ?
        """,
        (start_date, end_date),
    ).fetchone()

    actions = connection.execute(
        """
        SELECT action, COUNT(*) AS cnt
        FROM Audit_Log
        WHERE date(action_time) BETWEEN ? AND ?
        GROUP BY action
        """,
        (start_date, end_date),
    ).fetchall()

    return {
        "low_stock_threshold": LOW_STOCK_THRESHOLD,
        "low_stock_count": len(low_stock),
        "low_stock_items": [
            {"product_id": row["product_id"], "product_name": row["product_name"], "quantity": row["quantity"]}
            for row in low_stock
        ],
        "audit_events": int(audit["audit_events"]),
        "actions_by_type": {row["action"]: int(row["cnt"]) for row in actions},
    }


async def generate_sales_audit_report(payload: dict[str, Any] | None = None, context: Any = None) -> str:
    """Generate a sales audit report over a date window with progress notifications."""
    payload = payload or {}
    try:
        validated = GenerateSalesAuditReportInput.model_validate(payload)
    except ValidationError as exc:
        raise ToolValidationError(str(exc)) from exc

    try:
        start = date.fromisoformat(validated.start_date)
        end = date.fromisoformat(validated.end_date)
    except ValueError as exc:
        raise ToolValidationError("start_date and end_date must be valid ISO dates (YYYY-MM-DD)") from exc
    if start > end:
        raise ToolValidationError("start_date must be on or before end_date")

    await _send_progress(context, 0.00, "Initializing report generation...")
    await asyncio.sleep(PROGRESS_DELAY_SECONDS)

    with get_connection() as connection:
        await _send_progress(context, 0.25, "Aggregating total sales revenue...")
        sales = await _aggregate_sales(connection, validated.start_date, validated.end_date)
        await asyncio.sleep(PROGRESS_DELAY_SECONDS)

        await _send_progress(context, 0.50, "Calculating return ratios and refund metrics...")
        returns = await _aggregate_returns(connection, validated.start_date, validated.end_date)
        await asyncio.sleep(PROGRESS_DELAY_SECONDS)

        await _send_progress(context, 0.75, "Checking inventory levels and system audit logs...")
        inventory_and_audit = await _aggregate_inventory_and_audit(connection, validated.start_date, validated.end_date)
        await asyncio.sleep(PROGRESS_DELAY_SECONDS)

    order_count = sales["total_orders"]
    returns["return_rate"] = round(returns["total_returns"] / order_count, 4) if order_count else 0.0

    report = {
        "report_type": "sales_audit_report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "period": {"start_date": validated.start_date, "end_date": validated.end_date},
        "sales": sales,
        "returns": returns,
        "inventory": inventory_and_audit,
    }

    await _send_progress(context, 1.00, "Report complete.")
    return json.dumps(report, indent=2)


generate_sales_audit_report.name = "generate_sales_audit_report"
generate_sales_audit_report.kind = "tool"
