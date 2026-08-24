from __future__ import annotations

from state_graph.core.models import GraphState
from state_graph.graph1.graph import build_graph1
from state_graph.graph2.build import build_graph2
from state_graph.graph3.graph import build_graph3


def create_graph1(
    *,
    llm,
):
    """Create the Refund graph (LATS + HITL)."""
    return build_graph1(
        llm=llm,
    )


def create_graph2(
    *,
    llm,
):
    """
    Create the Shipping / Delivery Issue Investigation graph.

    Graph 2 uses Task Decomposition as Addition 1 and
    Constrained ReAct as Addition 2.
    """
    return build_graph2(
        llm=llm,
    )


def create_graph3(llm):
    return build_graph3(llm=llm)


def create_state(
    *,
    run_id: str,
    graph_name: str,
    goal: str,
    start_node: str = "awaiting_input",
) -> GraphState:
    return GraphState(
        run_id=run_id,
        graph_name=graph_name,
        current_node=start_node,
        goal=goal,
    )
