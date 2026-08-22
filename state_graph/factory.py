from __future__ import annotations

from planning_lab.algorithms.environment import Environment
from state_graph.core.models import GraphState


def create_graph1(
    environment: Environment | None = None,
):
    return build_graph1(
        environment=environment,
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
