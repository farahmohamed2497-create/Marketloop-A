"""Fixed, real-request prompts used by planning evaluations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SalesAuditCase:
    case_id: str
    prompt: str
    expected_action_fragment: str


PERSON2_CASES: tuple[SalesAuditCase, ...] = (
    SalesAuditCase(
        "lookahead-restock",
        "Analyze the January 2026 sales audit and choose the safest action for a low-stock product.",
        "restock",
    ),
    SalesAuditCase(
        "authorization-retry",
        "Propose a valid, manager-approved inventory action after the January 2026 sales audit.",
        "user_id",
    ),
    SalesAuditCase(
        "ungrounded-failure",
        "Select a restock action that is supported by the MarketLoop low-stock data, not intuition.",
        "product_id",
    ),
)