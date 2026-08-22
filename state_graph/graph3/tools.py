"""Tools for the Retention graph (Graph 3).

Three tools only, matching the problem-table entry exactly:
send_retention_offer, escalate_to_legal, apply_discount_code. The
Constrained ReAct node is given ONLY this dict — it cannot call anything
else, which is the whole point of "constrained" (it can't invent a
fourth action outside its delegated authority).

ASSUMPTION FLAGGED: schema below assumes a `Subscriptions` table
(subscription_id, customer_id, status, monthly_value, discount_pct) next
to the `Audit_Log` table already used by graph3's dispute tools. Adjust
column names to match the real MarketLoop schema before this ships —
that's the one thing I couldn't verify without DB access.

`apply_discount_code` is the only tool that actually mutates state (the
other two only propose / signal). Like `sync_dispute_resolution`, its two
writes (Subscriptions, then Audit_Log) are deliberately NOT one
transaction, for the same reason: they're two different write paths in
the real system, so a bare retry after a partial failure would either
re-apply the discount or double the audit entry — hence a ticket, not a
retry, on failure (handled generically by StateGraphEngine.step, not by
this graph's code).
"""
from __future__ import annotations

from typing import Any

from mcp_server.db import get_connection


class DiscountApplyError(ConnectionError):
    """Raised when the Audit_Log write for a discount application fails.

    Subclasses ConnectionError so classify_failure() in
    state_graph.core.exceptions files it under `tool_error`, not
    `unplanned_error` — same convention graph3's dispute tools use.
    """


def send_retention_offer(*, subscription_id: int, offer_type: str, offer_value: float) -> dict[str, Any]:
    """STUB. Propose a retention offer to the customer without applying it.

    Applying anything only ever happens through `apply_discount_code`,
    and only after the customer accepts (awaiting_customer_response) or
    an admin approves via HITL. This tool just records the proposal so
    the customer-facing side (email/SMS/in-app) can send it — replace
    the body with that real call.
    """
    return {
        "subscription_id": subscription_id,
        "offer_type": offer_type,
        "offer_value": offer_value,
        "proposed": True,
    }


def escalate_to_legal(*, reason: str) -> dict[str, Any]:
    """STUB tool the agent calls to signal a legal/compliance threat.

    Mirrors graph3's `escalate_to_hitl` convention: it only records the
    request inside the ReAct transcript. The actual pause + persistence
    happens in `RetentionGraph.retention_react` once the ReAct loop
    returns with `escalated=True` — this function must never itself talk
    to the HITL store, or the graph loses its single point of control
    over when a pause is durable.
    """
    return {"escalation_requested": True, "reason": reason}


def apply_discount_code(
    *,
    subscription_id: int,
    discount_pct: float,
    resolution_note: str,
    simulate_audit_failure: bool = False,
) -> dict[str, Any]:
    """Apply an approved discount to Subscriptions, then log it.

    `simulate_audit_failure` exists only so tests can deterministically
    trigger the Audit_Log write failing after the Subscriptions write
    already committed — same pattern as `sync_dispute_resolution`.
    """
    if not 0.0 < discount_pct <= 1.0:
        raise ValueError(f"discount_pct must be in (0, 1], got {discount_pct!r}")

    with get_connection() as connection:
        connection.execute(
            "UPDATE Subscriptions SET status = 'retained', discount_pct = ? WHERE subscription_id = ?",
            (discount_pct, subscription_id),
        )
        connection.commit()

    if simulate_audit_failure:
        raise DiscountApplyError(
            f"Audit_Log write failed after Subscriptions {subscription_id} "
            f"already set to discount_pct={discount_pct!r}."
        )

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO Audit_Log (action, table_name, record_id, details, user_id)
            VALUES ('discount_applied', 'Subscriptions', ?, ?, 1)
            """,
            (subscription_id, resolution_note),
        )
        connection.commit()

    return {"subscription_id": subscription_id, "discount_pct": discount_pct, "synced": True}