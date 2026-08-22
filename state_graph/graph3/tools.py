"""Tools for the escalated return-dispute graph (Graph 3).

`sync_dispute_resolution` performs two separate, non-transactional writes
against the same MarketLoop database: it updates Return_Requests, then
appends an Audit_Log entry. They are deliberately NOT wrapped in one SQL
transaction, because in production the audit trail is written by a
different internal write path than the return-status update. If the first
write succeeds and the second fails, blindly retrying the whole call would
re-apply the Return_Requests update (or produce a duplicate Audit_Log row)
instead of fixing the inconsistency -- which is exactly why this failure
needs a ticket, not an automatic retry.
"""

from __future__ import annotations

from typing import Any

from mcp_server.db import get_connection


class DisputeSyncError(ConnectionError):
    """Raised when the Audit_Log write for a dispute resolution fails.

    Subclasses ConnectionError so StateGraphEngine.step's failure
    classifier (`classify_failure`) files it under `tool_error` rather
    than `unplanned_error`.
    """


def check_return_dispute(*, return_id: int) -> dict[str, Any]:
    """Read the current Return_Requests + Orders context for a dispute."""
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

    return {
        "return_id": row["return_id"],
        "found": True,
        "status": row["status"],
        "reason": row["reason"],
        "order_total": row["total_amount"],
        "customer_name": row["customer_name"],
    }


def propose_retention_offer(*, return_id: int, offer_type: str, offer_value: float) -> dict[str, Any]:
    """STUB. Propose a retention offer without applying it.

    Replace the body with a real call to the discount/refund-adjustment
    service. The agent may only *propose* an offer here -- applying it
    happens exclusively through `sync_dispute_resolution`, after either
    the customer accepts or compliance approves.
    """
    return {
        "return_id": return_id,
        "offer_type": offer_type,
        "offer_value": offer_value,
        "proposed": True,
    }


def sync_dispute_resolution(
    *,
    return_id: int,
    decision: str,
    resolution_note: str,
    simulate_audit_failure: bool = False,
) -> dict[str, Any]:
    """Apply a dispute's final decision to Return_Requests, then Audit_Log.

    `simulate_audit_failure` exists only so tests can deterministically
    trigger the Audit_Log write failing after the Return_Requests write
    already committed, without needing to monkeypatch the database layer.
    """
    if decision not in {"Approved", "Rejected"}:
        raise ValueError(f"Unsupported dispute decision: {decision!r}")

    with get_connection() as connection:
        connection.execute(
            "UPDATE Return_Requests SET status = ? WHERE return_id = ?",
            (decision, return_id),
        )
        connection.commit()

    if simulate_audit_failure:
        raise DisputeSyncError(
            f"Audit_Log write failed after Return_Requests {return_id} "
            f"was already set to {decision!r}."
        )

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO Audit_Log (action, table_name, record_id, details, user_id)
            VALUES ('dispute_resolution', 'Return_Requests', ?, ?, 1)
            """,
            (return_id, resolution_note),
        )
        connection.commit()

    return {"return_id": return_id, "decision": decision, "synced": True}