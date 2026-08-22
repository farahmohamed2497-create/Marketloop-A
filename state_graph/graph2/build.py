from __future__ import annotations

from state_graph.core.engine import StateGraphEngine
from state_graph.core.models import GraphState
from .design import shipping_transition_table
from .shipping_graph import ShippingGraph


def build_graph2(*, llm) -> StateGraphEngine:
    graph = ShippingGraph(llm=llm)

    return StateGraphEngine(
        transitions=shipping_transition_table(),
        nodes=graph.nodes(),
    )


def create_initial_state(run_id: str, goal: str) -> GraphState:
    return GraphState(
        run_id=run_id,
        graph_name="shipping",
        current_node="awaiting_input",
        goal=goal,
    )
