"""
Promote-or-drop router (Concern: "Promote-or-drop routing, forget and
episodic only").

Fires when the rolling buffer (rolling_buffer.py) overflows and an item
is about to age out. For each aging item, decides:
  - FORGET: safe to drop, it's re-fetchable or was only ever noise
  - PROMOTE: worth keeping in episodic memory beyond this session

Real MarketLoop need: when a support call ends, most of the transcript
is tool noise (order lookups, shipment checks) that's cheap to re-fetch
next time. But a handful of turns are worth remembering across sessions
- e.g. a customer's return reason on a still-open dispute, or a pattern
worth flagging (repeat damaged-item complaints). Losing those means the
next agent who picks up the case starts blind; keeping *everything*
means episodic memory fills with re-fetchable noise.

Hard rule from the lab spec: this router NEVER writes to semantic
memory directly. It only ever chooses FORGET or PROMOTE-TO-EPISODIC.
Semantic memory is built later by a separate, periodic consolidation
pass over the episodic store (a teammate's task, topic 10) - not here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .episodic_store import EpisodicMemory, EpisodicStore

# Patterns that make a user-stated fact worth remembering past this
# session. Kept narrow and MarketLoop-specific on purpose: a router that
# promotes everything "just in case" isn't actually making a decision.
DURABLE_FACT_PATTERNS = [
    r"\breturn reason\b",
    r"\bdamaged\b",
    r"\bwrong item\b",
    r"\bchanged their mind\b",
    r"\bno longer needs\b",
    r"\baddress\b",
    r"\bcomplain(t|ed|ing)?\b",
    r"\ballerg",
    r"\bvip\b",
    r"\bloyalty\b",
]

_DURABLE_RE = re.compile("|".join(DURABLE_FACT_PATTERNS), re.IGNORECASE)


@dataclass
class RoutingDecision:
    content: str
    role: str
    decision: str  # "forget" | "promote"
    reasoning: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class PromoteDropRouter:
    def __init__(self) -> None:
        # Shared episodic store used by the consolidation layer.
        self.episodic_store = EpisodicStore()

        # Log kept only for grading/debugging.
        self.decision_log: list[RoutingDecision] = []


    def _decide(self, item: dict[str, Any]) -> RoutingDecision:
        role = item.get("role", "")
        content = item.get("content", "")

        if role == "tool":
            return RoutingDecision(
                content=content,
                role=role,
                decision="forget",
                reasoning=(
                    "Raw tool output - re-fetchable on demand from the "
                    "same MCP tool, no reason to keep it beyond this session."
                ),
            )

        if role == "system":
            return RoutingDecision(
                content=content,
                role=role,
                decision="forget",
                reasoning=(
                    "Derived summary text, not a primary source - keeping "
                    "it in episodic memory would duplicate whatever "
                    "generated it."
                ),
            )

        if role == "user" and _DURABLE_RE.search(content):
            return RoutingDecision(
                content=content,
                role=role,
                decision="promote",
                reasoning=(
                    "Customer-stated fact matching a durable-fact pattern "
                    "(e.g. return reason, complaint, contact change) - "
                    "relevant if this case is revisited in a future session."
                ),
            )

        return RoutingDecision(
            content=content,
            role=role,
            decision="forget",
            reasoning=(
                "User turn with no durable-fact signal (routine chat/"
                "confirmation) - safe to drop once this session ends."
            ),
        )

    def route(
        self,
        aging_items: list[dict[str, Any]],
    ) -> list[RoutingDecision]:
        """
        Called with the item(s) about to be evicted from the rolling
        buffer. Returns the decision for each, and stores PROMOTE
        decisions in the episodic store.
        """
        decisions: list[RoutingDecision] = []

        for item in aging_items:
            decision = self._decide(item)

            self.decision_log.append(decision)

            if decision.decision == "promote":
                self.episodic_store.add(
                    EpisodicMemory(
                        content=decision.content,
                        role=decision.role,
                        reasoning=decision.reasoning,
                        timestamp=decision.timestamp,
                    )
                )

            decisions.append(decision)

        return decisions

    def get_reasoning_log(self) -> list[dict[str, str]]:
        """Grader-visible log of every routing decision."""
        return [
            {
                "content": d.content,
                "role": d.role,
                "decision": d.decision,
                "reasoning": d.reasoning,
                "timestamp": d.timestamp,
            }
            for d in self.decision_log
        ]
