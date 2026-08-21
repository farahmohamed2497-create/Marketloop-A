from __future__ import annotations

from state_graph.core.engine import StateGraphEngine
from state_graph.core.models import GraphState
from state_graph.core.transitions import TransitionTable

from .shipping_graph import ShippingGraph


def build_graph2(*, llm) -> StateGraphEngine:
    graph = ShippingGraph(llm=llm)

    transitions = TransitionTable()

    transitions.add("awaiting_input", "decompose")
    transitions.add("decompose", "constrained_react")
    transitions.add("constrained_react", "done")

    return StateGraphEngine(
        transitions=transitions,
        nodes=graph.nodes(),
    )


def create_initial_state(run_id: str, goal: str) -> GraphState:
    return GraphState(
        run_id=run_id,
        graph_name="shipping",
        current_node="awaiting_input",
        goal=goal,
    )
