from __future__ import annotations

from typing import Any


def requires_human_intervention(
    *,
    score: float | None = None,
    refund_amount: float | None = None,
    policy_violation: bool = False,
    confidence_threshold: float = 0.70,
    amount_threshold: float = 500.0,
) -> bool:
    """
    Decide whether a refund request must be escalated to a human.

    HITL is required when:
    - the model confidence is below the configured threshold,
    - the refund amount exceeds the configured approval limit, or
    - the proposed action violates the refund policy.
    """

    if score is not None and score < confidence_threshold:
        return True

    if refund_amount is not None and refund_amount > amount_threshold:
        return True

    if policy_violation:
        return True

    return False