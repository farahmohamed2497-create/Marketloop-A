"""Graph 3 — Retention: state enum + node registry.

Ticket item: `feat(graph3): define State enum and node registry`

`RetentionNode` enumerates every node this graph can be parked at in
`GraphState.current_node`. It intentionally does NOT include a member for
"waiting on HITL" or "waiting on the customer" as separate node names —
those are represented by `GraphState.status == "waiting"` while
`current_node` stays at whichever node is doing the waiting
(`retention_react` for HITL, `awaiting_customer_response` for the
customer). That mirrors how `StateGraphEngine.step` already treats a
result whose `next_node == current node` as a no-op transition needing no
entry in the `TransitionTable` (see engine.py: validation is skipped when
`result.next_node != node`). Two separate "waiting" node names would just
duplicate that mechanism.
"""
from __future__ import annotations

from enum import Enum



from typing import TYPE_CHECKING

from state_graph.core.engine import NodeFunction

if TYPE_CHECKING:
    from .retention_graph import RetentionGraph
    
class RetentionNode(str, Enum):
    """Every node Graph 3 (Retention) can be parked at."""

    AWAITING_INPUT = "awaiting_input"
    POLICY_LOOKUP = "policy_lookup"                    # LLM addition #2 (RAG) — teammate
    RETENTION_REACT = "retention_react"                 # LLM addition #1 (Constrained ReAct)
    AWAITING_CUSTOMER_RESPONSE = "awaiting_customer_response"
    DONE = "done"


def build_node_registry(graph: "RetentionGraph") -> dict[str, "NodeFunction"]:  # noqa: F821
    """Single source of truth mapping RetentionNode -> bound node method.

    Both `graph.py` (to construct the StateGraphEngine) and the test
    suite (to call transition functions in isolation without duplicating
    the wiring) import this instead of re-listing the methods.
    """
    return {
        RetentionNode.AWAITING_INPUT.value: graph.awaiting_input,
        RetentionNode.POLICY_LOOKUP.value: graph.policy_lookup,
        RetentionNode.RETENTION_REACT.value: graph.retention_react,
        RetentionNode.AWAITING_CUSTOMER_RESPONSE.value: graph.awaiting_customer_response,
    }