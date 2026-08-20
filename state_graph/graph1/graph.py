from planning_lab.algorithms.environment import Environment

from state_graph.core.engine import StateGraphEngine
from state_graph.core.models import GraphState
from state_graph.core.transitions import TransitionTable

from .refund_graph import RefundGraph


def build_graph1(
    *,
    llm,
) -> StateGraphEngine:

    environment = Environment()

    graph = RefundGraph(
        llm=llm,
        environment=environment,
    )

    transitions = TransitionTable()

    transitions.add(
        "awaiting_input",
        "lats",
    )

    return StateGraphEngine(
        transitions=transitions,
        nodes=graph.nodes(),
    )


def create_initial_state(
    run_id: str,
    goal: str,
) -> GraphState:

    return GraphState(
        run_id=run_id,
        graph_name="refund",
        current_node="awaiting_input",
        goal=goal,
    )