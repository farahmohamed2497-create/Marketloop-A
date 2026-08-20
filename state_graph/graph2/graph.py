from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from planning_lab.algorithms.environment import Environment

from state_graph.core.engine import StateGraphEngine
from state_graph.core.models import GraphState
from state_graph.core.transitions import TransitionTable

from .lats_hitl import LATSAndHITLGraph


def build_graph2(
    *,
    llm: BaseChatModel,
    environment: Environment,
    confidence_threshold: float = 0.70,
) -> StateGraphEngine:

    graph = LATSAndHITLGraph(
        llm=llm,
        environment=environment,
        confidence_threshold=confidence_threshold,
    )

    transitions = TransitionTable()

    transitions.add(
        "awaiting_input",
        "lats_search",
    )

    # HITL is a waiting state.
    # The external resume operation moves it to "resume".
    transitions.add(
        "hitl",
        "resume",
    )

    transitions.add(
        "resume",
        "lats_search",
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
        graph_name="lats_hitl",
        current_node="awaiting_input",
        goal=goal,
    )