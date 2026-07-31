"""Defensive order-related MCP tools."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError

from ..db import get_connection


class ElicitationContext:
    """Minimal context object carrying an MCP session for elicitation."""

    def __init__(self, session: Any | None = None) -> None:
        self.session = session


class ProcessReturnRequestInput(BaseModel):
    """Strict input schema for processing a return request."""

    order_id: int = Field(..., ge=1)
    customer_id: int = Field(..., ge=1)
    reason: str = Field(..., min_length=5)

    model_config = {
        "extra": "forbid",
    }


class ToolValidationError(ValueError):
    """Raised when the input or business rules are invalid."""


class AuthorizationError(PermissionError):
    """Raised when the caller is not authorized to perform the action."""


async def process_return_request(payload: dict[str, Any], context: ElicitationContext | None = None) -> dict[str, Any]:
    """Create a return request for a delivered order only, with elicitation for high-value requests."""
    try:
        validated = ProcessReturnRequestInput.model_validate(payload)
    except ValidationError as exc:
        raise ToolValidationError(str(exc)) from exc

    with get_connection() as connection:
        order_row = connection.execute(
            "SELECT status, customer_id, total_amount FROM Orders WHERE order_id = ?",
            (validated.order_id,),
        ).fetchone()
        if order_row is None:
            raise ToolValidationError("Order not found")

        if order_row["status"] in {"Pending", "Processing", "Shipped"}:
            raise ToolValidationError(
                "Return requests can only be created for delivered orders"
            )

        if order_row["customer_id"] != validated.customer_id:
            raise ToolValidationError("Order does not belong to the specified customer")

        if order_row["status"] != "Delivered":
            raise ToolValidationError("Order must be delivered before a return can be requested")

        total_amount = float(order_row["total_amount"] or 0.0)
        if total_amount > 100.00:
            session = context.session if context is not None else None
            if session is None or not hasattr(session, "elicit_form"):
                raise ToolValidationError("Elicitation is unavailable for high-value returns")

            result = await session.elicit_form(
                f"High-value return detected (${total_amount:.2f}). Do you authorize creating a return request for Order #{validated.order_id}?",
                {"type": "object", "properties": {"approved": {"type": "boolean"}}},
                related_request_id=None,
            )
            if getattr(result, "action", None) != "accept":
                return {
                    "status": "cancelled",
                    "message": "Return request cancelled by administrative intervention.",
                    "order_id": validated.order_id,
                }

        cursor = connection.execute(
            """
            INSERT INTO Return_Requests (reason, request_date, status, order_id, customer_id)
            VALUES (?, date('now'), 'Pending', ?, ?)
            """,
            (validated.reason, validated.order_id, validated.customer_id),
        )
        connection.commit()
        return {
            "status": "Pending",
            "return_id": cursor.lastrowid,
            "order_id": validated.order_id,
            "customer_id": validated.customer_id,
        }
