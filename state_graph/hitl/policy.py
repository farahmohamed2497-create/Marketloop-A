from __future__ import annotations


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


def shipping_requires_human_intervention(
    *,
    confidence: float | None = None,
    claim_amount: float | None = None,
    policy_violation: bool = False,
    confidence_threshold: float = 0.70,
    amount_threshold: float = 500.0,
) -> bool:
    """
    Decide whether a shipping/delivery resolution must be escalated
    to a human.

    Thresholds are kept identical to the refund graph's, per project
    convention, so admins see one consistent bar across graphs rather
    than a different one per domain.

    HITL is required when:
    - the constrained-ReAct confidence is below the configured
      threshold (the agent needed several tool calls to resolve the
      issue, or asked to escalate itself),
    - the carrier claim amount exceeds the configured approval
      limit, or
    - the proposed action contradicts stated policy (e.g. filing a
      carrier claim against a shipment tracking already reports as
      delivered).
    """

    if confidence is not None and confidence < confidence_threshold:
        return True

    if claim_amount is not None and claim_amount > amount_threshold:
        return True

    if policy_violation:
        return True

    return False


def inventory_requires_human_intervention(
    *,
    confidence: float | None = None,
    quantity_variance: int | None = None,
    policy_violation: bool = False,
    confidence_threshold: float = 0.70,
    variance_threshold: int = 10,
) -> bool:
    """Require an admin before an inventory discrepancy changes stock."""

    return (
        (confidence is not None and confidence < confidence_threshold)
        or (quantity_variance is not None and abs(quantity_variance) >= variance_threshold)
        or policy_violation
    )

def dispute_requires_compliance_review(
    *,
    confidence: float | None = None,
    retention_offer_value: float | None = None,
    legal_threat: bool = False,
    policy_violation: bool = False,
    confidence_threshold: float = 0.70,
    offer_value_threshold: float = 500.0,
) -> bool:
    """Require a compliance admin before a retention offer is finalized.

    HITL is required when:
    - the constrained-ReAct confidence is below the configured threshold,
    - the proposed retention offer exceeds the configured approval limit,
    - the customer has raised an explicit legal / chargeback threat, or
    - the proposed action contradicts stated policy (e.g. offering a
      refund on an already-rejected, non-defective return).
    """

    return (
        (confidence is not None and confidence < confidence_threshold)
        or (retention_offer_value is not None and retention_offer_value > offer_value_threshold)
        or legal_threat
        or policy_violation
    )