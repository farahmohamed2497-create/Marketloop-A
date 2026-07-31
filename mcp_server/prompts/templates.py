"""Prompt templates exposed by the MarketLoop MCP server."""

from __future__ import annotations

from typing import Any

DRAFT_RETURN_RESPONSE = """Subject: Update on Your Return Request for Order #{order_id}

Dear {customer_name},

Thank you for contacting MarketLoop Customer Support regarding your return request for order #{order_id}.

We have reviewed your request and the decision is: **{decision_status}**.

{explanation}

If you have any further questions, please reply to this email or reach out to our support team and we will be happy to help.

Best regards,
MarketLoop Customer Support Team
"""

_APPROVED_EXPLANATION = (
    "Your return has been approved. A prepaid return label will be sent to you within 24 hours. "
    "Please ship the item back within 14 days of receiving the label. "
    "Once the item arrives and passes inspection at our warehouse, your refund will be processed "
    "within 5-10 business days to your original payment method. Restocking fees may apply per our "
    "return and refund policy."
)

_REJECTED_EXPLANATION = (
    "After careful review, we are unable to approve this return request for the following reason: "
    "{reason}. If you believe this decision was made in error, or if you have additional information "
    "you would like us to consider, please reply to this email and we will re-open your case."
)


def product_prompt(arguments: dict[str, Any] | None = None) -> str:
    """Example prompt placeholder."""
    return "product prompt"


def draft_return_response_prompt(arguments: dict[str, Any] | None = None) -> str:
    """Draft a customer service email responding to a return request decision."""
    args = arguments or {}
    required = ("order_id", "customer_name", "decision_status", "reason")
    missing = [name for name in required if name not in args]
    if missing:
        raise ValueError(f"Missing required prompt arguments: {', '.join(missing)}")

    order_id = str(args["order_id"]).strip()
    customer_name = str(args["customer_name"]).strip()
    decision_status = str(args["decision_status"]).strip()
    reason = str(args["reason"]).strip()
    if not order_id or not customer_name or not reason:
        raise ValueError("Prompt arguments order_id, customer_name, and reason must not be empty")

    normalized_status = decision_status.lower()
    if normalized_status not in {"approved", "rejected"}:
        raise ValueError("decision_status must be either 'Approved' or 'Rejected'")
    decision_status = "Approved" if normalized_status == "approved" else "Rejected"

    explanation = (
        _APPROVED_EXPLANATION
        if normalized_status == "approved"
        else _REJECTED_EXPLANATION.format(reason=reason)
    )

    return DRAFT_RETURN_RESPONSE.format(
        order_id=order_id,
        customer_name=customer_name,
        decision_status=decision_status,
        explanation=explanation,
    )


draft_return_response_prompt.name = "draft_return_response"
draft_return_response_prompt.kind = "prompt"
draft_return_response_prompt.arguments = [
    {
        "name": "order_id",
        "description": "The order number for the return request.",
        "required": True,
    },
    {
        "name": "customer_name",
        "description": "The name of the customer to address the email to.",
        "required": True,
    },
    {
        "name": "decision_status",
        "description": "The return decision: 'Approved' or 'Rejected'.",
        "required": True,
    },
    {
        "name": "reason",
        "description": "The reason for the decision (used when the return is rejected).",
        "required": True,
    },
]
