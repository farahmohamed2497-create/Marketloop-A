"""
The long-context test suite for MarketLoop's memory strategies.

Real problem: a support agent processes a return request. The customer
states the return reason in turn 1 (e.g. "item arrived damaged" vs.
"changed my mind"). Per MarketLoop's return policy, this single fact
decides whether a 15% restocking fee applies. Between turn 1 and the
final decision, the agent fires 30+ tool calls (order lookup, shipment
tracking, inventory checks, fee lookups) - realistic noise for a call
this length. If the strategy loses the reason, the agent risks charging
a customer for damage that was not their fault.

Per the lab's guardrail: keep this test suite FIXED once evaluation
starts. Do not edit CASES after you begin recording results.
"""

from __future__ import annotations

import random
from typing import Any

TOOL_CALL_TEMPLATES = [
    "get_order_details(order_id={oid})",
    "get_customer_profile(customer_id={cid})",
    "check_shipment_status(tracking='EG{tracking}')",
    "get_product_details(product_id={pid})",
    "check_inventory(product_id={pid})",
    "get_return_policy_section(topic='restocking_fee')",
    "list_order_items(order_id={oid})",
    "get_discount_history(product_id={pid})",
]

REASONS = [
    ("item arrived damaged in shipping", True),   # (reason, no_fee_expected)
    ("wrong item was shipped", True),
    ("customer changed their mind", False),
    ("customer no longer needs the item", False),
]


def _tool_noise(n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        template = rng.choice(TOOL_CALL_TEMPLATES)
        call = template.format(
            oid=rng.randint(1000, 9999),
            cid=rng.randint(100, 999),
            pid=rng.randint(10, 99),
            tracking=rng.randint(1000000, 9999999),
        )
        out.append({"role": "tool", "content": f"[{call}] -> result_{i}: OK"})
    return out


def build_case(reason: str, no_fee_expected: bool, noise_turns: int, seed: int) -> dict[str, Any]:
    messages = [
        {"role": "user", "content": f"Customer wants to return their laptop. Reason: {reason}."}
    ]
    messages += _tool_noise(noise_turns, seed)
    messages.append(
        {"role": "user", "content": "Should a 15% restocking fee apply to this return?"}
    )
    return {
        "messages": messages,
        "reason": reason,
        "no_fee_expected": no_fee_expected,
    }


def build_test_suite(variations_per_reason: int = 3, noise_turns: int = 35) -> list[dict[str, Any]]:
    """10 variations, matching the worked example's scale, across all 4
    reason types with different random tool-call noise each time."""
    cases = []
    seed = 0
    for reason, no_fee_expected in REASONS:
        for _ in range(variations_per_reason):
            cases.append(build_case(reason, no_fee_expected, noise_turns, seed))
            seed += 1
    return cases


def reason_survived(messages: list[dict[str, Any]], reason: str) -> bool:
    """Does the exact return reason still appear in what's left of the
    transcript after a pruning strategy has run?"""
    text = " ".join(m.get("content", "") for m in messages)
    return reason in text
