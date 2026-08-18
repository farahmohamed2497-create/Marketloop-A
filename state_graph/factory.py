from __future__ import annotations

from planning_lab.algorithms.environment import Environment

from state_graph.core.models import GraphState

from .graph1.graph import build_graph1
from .graph2.graph import build_graph2
from .graph3.graph import build_graph3


def create_graph1(
    environment: Environment | None = None,
):
    return build_graph1(
        environment=environment,
    )


def create_graph2(
    environment: Environment,
    confidence_threshold: float = 0.70,
):
    return build_graph2(
        environment=environment,
        confidence_threshold=confidence_threshold,
    )


def create_graph3(
    base_engine,
):
    return build_graph3(
        base_engine,
    )


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