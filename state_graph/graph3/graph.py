"""Graph 3 — Retention: edge wiring.

Ticket item: `feat(graph3): wire graph edges (cycles + conditional branches)`

Edge list is deliberately short — five entries — because most of the
branching in this graph is CONDITIONAL, decided at runtime inside
`retention_react` and `awaiting_customer_response` (which of several
already-allowed targets to take), not encoded as extra edges. The two
genuine CYCLES are:

  awaiting_customer_response -> retention_react
      A customer rejection with negotiation rounds remaining sends the
      graph back into retention_react for another offer. The same node
      is revisited more than once per run with new information each
      time -- see README section "why this graph needs cycles."

  retention_react -> retention_react   (implicit, not a table entry)
      HITL pause/resume: StateGraphEngine.step skips TransitionTable
      validation whenever `next_node == current node`, so a self-loop
      needs no explicit edge. This is how retention_react can pause for
      compliance review and, once an admin decides, resume into the
      SAME node to apply that decision.
"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from state_graph.core.engine import StateGraphEngine
from state_graph.core.models import GraphState
from state_graph.core.transitions import TransitionTable

from .retention_graph import RetentionGraph
from .state import RetentionNode


def build_graph3(*, llm: BaseChatModel) -> StateGraphEngine:
    graph = RetentionGraph(llm=llm)

    transitions = TransitionTable()

    transitions.add(RetentionNode.AWAITING_INPUT.value, RetentionNode.POLICY_LOOKUP.value)
    transitions.add(RetentionNode.POLICY_LOOKUP.value, RetentionNode.RETENTION_REACT.value)
    transitions.add(RetentionNode.RETENTION_REACT.value, RetentionNode.AWAITING_CUSTOMER_RESPONSE.value)
    transitions.add(RetentionNode.RETENTION_REACT.value, RetentionNode.DONE.value)
    transitions.add(RetentionNode.AWAITING_CUSTOMER_RESPONSE.value, RetentionNode.RETENTION_REACT.value)

    return StateGraphEngine(
        transitions=transitions,
        nodes=graph.nodes(),
    )


def create_initial_state(
    *,
    run_id: str,
    goal: str,
    subscription_id: int,
    customer_id: int | None = None,
    retention_offer_value: float | None = None,
    legal_threat: bool = False,
    policy_violation: bool = False,
    proposed_action: str = "send_retention_offer",
) -> GraphState:
    return GraphState(
        run_id=run_id,
        graph_name="retention",
        current_node=RetentionNode.AWAITING_INPUT.value,
        goal=goal,
        data={
            "subscription_id": subscription_id,
            "customer_id": customer_id,
            "retention_offer_value": retention_offer_value,
            "legal_threat": legal_threat,
            "policy_violation": policy_violation,
            "proposed_action": proposed_action,
            "negotiation_round": 0,
        },
    )