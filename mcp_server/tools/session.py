"""Session-management tools and dynamic tool exposure for MarketLoop."""

from __future__ import annotations

from typing import Any

from ..db import get_connection


class SessionContext:
    """Tracks the active role for the current server connection."""

    def __init__(self) -> None:
        self.active_role: str | None = None


class ToolRegistryError(RuntimeError):
    """Raised when the tool registry cannot be updated."""


def switch_active_user_role(user_id: int, session_context: SessionContext | None = None) -> dict[str, Any]:
    """Switch the effective role for the current session context and expose tools accordingly."""
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT u.user_id, r.role_name
            FROM Users AS u
            JOIN Roles AS r ON u.role_id = r.role_id
            WHERE u.user_id = ?
            """,
            (user_id,),
        ).fetchone()
        if row is None:
            raise ToolRegistryError("User not found")

        new_role = (row["role_name"] or "").strip()
        if session_context is not None:
            previous_role = session_context.active_role
            session_context.active_role = new_role
            if previous_role != new_role:
                return {
                    "status": "role_changed",
                    "user_id": user_id,
                    "active_role": new_role,
                    "tools_updated": True,
                    "notification": "notifications/tools/list_changed",
                }

        return {
            "status": "role_unchanged",
            "user_id": user_id,
            "active_role": new_role,
            "tools_updated": False,
            "notification": None,
        }
