"""Graph 3 wiring for an escalated, already-closed return dispute."""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from state_graph.core.engine import StateGraphEngine
from state_graph.core.models import GraphState
from state_graph.core.transitions import TransitionTable

from .dispute_graph import DisputeGraph


def build_graph3(*, llm: BaseChatModel) -> StateGraphEngine:
    graph = DisputeGraph(llm=llm)
    transitions = TransitionTable()
    transitions.add("awaiting_input", "retention_strategy")
    transitions.add("retention_strategy", "dispute_react")
    transitions.add("dispute_react", "awaiting_customer_response")
    transitions.add("dispute_react", "done")
    transitions.add("awaiting_customer_response", "dispute_react")
    return StateGraphEngine(transitions=transitions, nodes=graph.nodes())


def create_initial_state(*, run_id: str, goal: str, return_id: int,
                         retention_offer_value: float = 0, legal_threat: bool = False,
                         policy_violation: bool = False) -> GraphState:
    return GraphState(
        run_id=run_id, graph_name="dispute", current_node="awaiting_input", goal=goal,
        data={"return_id": return_id, "retention_offer_value": retention_offer_value,
              "legal_threat": legal_threat, "policy_violation": policy_violation},
    )
