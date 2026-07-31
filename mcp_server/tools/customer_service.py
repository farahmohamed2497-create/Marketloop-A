"""Customer service tools that use MCP sampling to generate responses."""

from __future__ import annotations

from typing import Any

from mcp import types
from pydantic import BaseModel, Field, ValidationError

from ..db import get_connection


class GenerateDelayApologyInput(BaseModel):
    """Strict input schema for generating a delay apology email."""

    order_id: int = Field(..., ge=1, description="The order number to apologize for.")

    model_config = {
        "extra": "forbid",
    }


class ToolValidationError(ValueError):
    """Raised when the input or business rules are invalid."""


class SamplingUnavailableError(RuntimeError):
    """Raised when the connected client does not support MCP sampling."""


_DELAY_APOLOGY_SYSTEM_PROMPT = (
    "You are a customer service writer for MarketLoop, an Egyptian e-commerce company. "
    "Write a short, warm, and professional email apologizing to a customer for a delayed "
    "shipment. Keep it under 150 words, be empathetic, acknowledge the inconvenience, and "
    "reassure the customer that MarketLoop is resolving the issue. Mention the order number "
    "and the customer's name."
)


async def generate_delay_apology(payload: dict[str, Any] | None = None, context: Any = None) -> str:
    """Generate a personalized shipment-delay apology email using MCP sampling."""
    payload = payload or {}
    try:
        validated = GenerateDelayApologyInput.model_validate(payload)
    except ValidationError as exc:
        raise ToolValidationError(str(exc)) from exc

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT o.order_id, o.order_date, o.status, c.name AS customer_name
            FROM Orders AS o
            JOIN Customers AS c ON o.customer_id = c.customer_id
            WHERE o.order_id = ?
            """,
            (validated.order_id,),
        ).fetchone()
    if row is None:
        raise ToolValidationError("Order not found")

    session = getattr(context, "session", None) if context is not None else None
    if session is None or not hasattr(session, "create_message"):
        raise SamplingUnavailableError(
            "The connected client does not support MCP sampling; cannot generate an apology email."
        )

    user_context = (
        f"Customer name: {row['customer_name']}\n"
        f"Order number: {row['order_id']}\n"
        f"Order date: {row['order_date']}\n"
        f"Current order status: {row['status']}\n\n"
        "Write the apology email addressed to this customer."
    )

    related_request_id = getattr(context, "request_id", None) if context is not None else None
    result = await session.create_message(
        messages=[
            types.SamplingMessage(
                role="user",
                content=types.TextContent(type="text", text=user_context),
            )
        ],
        system_prompt=_DELAY_APOLOGY_SYSTEM_PROMPT,
        max_tokens=300,
        temperature=0.7,
        related_request_id=related_request_id,
    )
    return result.content.text


generate_delay_apology.name = "generate_delay_apology"
generate_delay_apology.kind = "tool"
