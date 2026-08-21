from __future__ import annotations

from typing import Any


def check_tracking(*, tracking_number: str) -> dict[str, Any]:
    """
    STUB. Look up the current status of a shipment.

    Replace the body with a real call to the carrier's tracking API.
    Kept deliberately deterministic here so tests are reproducible.
    """
    return {
        "tracking_number": tracking_number,
        "status": "in_transit",
        "last_update": "stub-not-implemented",
    }


def open_carrier_claim(
    *,
    tracking_number: str,
    reason: str,
    claim_amount: float,
) -> dict[str, Any]:
    """
    STUB. File a claim with the carrier for a lost/damaged shipment.

    Replace the body with a real call to the carrier's claims API.
    """
    return {
        "tracking_number": tracking_number,
        "claim_id": "stub-claim-id",
        "reason": reason,
        "claim_amount": claim_amount,
        "status": "submitted",
    }


def escalate_to_hitl(*, reason: str) -> dict[str, Any]:
    """
    STUB tool the agent calls to signal it cannot resolve the issue
    on its own.

    IMPORTANT: this tool only records the agent's own request inside
    the ReAct transcript. It does NOT itself pause the graph or write
    to HITL_Requests — that pause + persistence is handled by
    HITLNode.pause() back in ShippingGraph.constrained_react_node,
    once the ReAct loop has returned. Two separate concerns: the
    model deciding it's stuck, and the graph engine actually pausing.
    """
    return {
        "escalation_requested": True,
        "reason": reason,
    }
