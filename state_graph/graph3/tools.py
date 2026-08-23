"""Whitelisted tools for Graph 3's escalated return-dispute workflow."""
from __future__ import annotations

from typing import Any

from mcp_server.db import get_connection


class DisputeSyncError(ConnectionError):
    """A partial dispute write that must be resolved by a failure ticket."""


def check_return_dispute(*, return_id: int) -> dict[str, Any]:
    """Return the closed-return evidence that the LLM may use."""
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT r.return_id, r.reason, r.status, r.request_date,
                   o.order_id, o.total_amount, c.name AS customer_name
            FROM Return_Requests AS r
            JOIN Orders AS o ON r.order_id = o.order_id
            JOIN Customers AS c ON r.customer_id = c.customer_id
            WHERE r.return_id = ?
            """,
            (return_id,),
        ).fetchone()
    if row is None:
        return {"return_id": return_id, "found": False}
    return {"return_id": row["return_id"], "found": True, "status": row["status"],
            "reason": row["reason"], "order_total": row["total_amount"],
            "customer_name": row["customer_name"]}


def propose_retention_offer(*, return_id: int, offer_type: str, offer_value: float) -> dict[str, Any]:
    """Record a proposal only; finalization remains customer/HITL controlled."""
    return {"return_id": return_id, "offer_type": offer_type, "offer_value": offer_value, "proposed": True}


def sync_dispute_resolution(*, return_id: int, decision: str, resolution_note: str,
                            simulate_audit_failure: bool = False) -> dict[str, Any]:
    """Write the decision then its audit record; a partial write is not retried."""
    if decision not in {"Approved", "Rejected"}:
        raise ValueError(f"Unsupported dispute decision: {decision!r}")
    with get_connection() as connection:
        connection.execute("UPDATE Return_Requests SET status = ? WHERE return_id = ?", (decision, return_id))
        connection.commit()
    if simulate_audit_failure:
        raise DisputeSyncError(f"Audit_Log write failed after Return_Requests {return_id} was set to {decision!r}.")
    with get_connection() as connection:
        connection.execute(
            "INSERT INTO Audit_Log (action, table_name, record_id, details, user_id) VALUES ('dispute_resolution', 'Return_Requests', ?, ?, 1)",
            (return_id, resolution_note),
        )
        connection.commit()
    return {"return_id": return_id, "decision": decision, "synced": True}
