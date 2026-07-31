"""Defensive inventory-related MCP tools."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError

from ..db import get_connection


class UpdateInventoryQuantityInput(BaseModel):
    """Strict input schema for updating inventory quantity."""

    product_id: int = Field(..., ge=1)
    quantity_change: int = Field(...)
    user_id: int = Field(..., ge=1)

    model_config = {
        "extra": "forbid",
    }


class ToolValidationError(ValueError):
    """Raised when the input or business rules are invalid."""


class AuthorizationError(PermissionError):
    """Raised when the caller is not authorized to perform the action."""


def inventory_tool() -> str:
    """Example inventory tool placeholder."""
    return "inventory tool"


def update_inventory_quantity(payload: dict[str, Any]) -> dict[str, Any]:
    """Update inventory quantity for authorized warehouse or management users."""
    try:
        validated = UpdateInventoryQuantityInput.model_validate(payload)
    except ValidationError as exc:
        raise ToolValidationError(str(exc)) from exc

    with get_connection() as connection:
        user_row = connection.execute(
            """
            SELECT u.user_id, r.role_name
            FROM Users AS u
            JOIN Roles AS r ON u.role_id = r.role_id
            WHERE u.user_id = ?
            """,
            (validated.user_id,),
        ).fetchone()
        if user_row is None:
            raise AuthorizationError("User not found")

        role_name = (user_row["role_name"] or "").strip().lower()
        if role_name not in {"warehouse admin", "manager", "inventory manager"}:
            raise AuthorizationError("User is not authorized to update inventory")

        inventory_row = connection.execute(
            "SELECT quantity FROM Inventory WHERE product_id = ?",
            (validated.product_id,),
        ).fetchone()
        if inventory_row is None:
            raise ToolValidationError("Inventory record not found for the specified product")

        new_quantity = inventory_row["quantity"] + validated.quantity_change
        connection.execute(
            "UPDATE Inventory SET quantity = ?, last_updated = CURRENT_TIMESTAMP WHERE product_id = ?",
            (new_quantity, validated.product_id),
        )
        connection.execute(
            """
            INSERT INTO Audit_Log (action, table_name, record_id, details, user_id)
            VALUES (?, 'Inventory', ?, ?, ?)
            """,
            (
                "UPDATE_INVENTORY",
                validated.product_id,
                f"quantity adjusted by {validated.quantity_change}",
                validated.user_id,
            ),
        )
        connection.commit()
        return {
            "status": "updated",
            "product_id": validated.product_id,
            "new_quantity": new_quantity,
            "user_id": validated.user_id,
        }
