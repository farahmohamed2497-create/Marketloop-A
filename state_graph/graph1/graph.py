from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from state_graph.core.engine import StateGraphEngine
from state_graph.core.models import GraphState
from state_graph.core.transitions import TransitionTable

from .decomposition_execution import (
    DecompositionExecutionGraph,
)


def build_graph1(
    *,
    llm: BaseChatModel,
) -> StateGraphEngine:

    graph = DecompositionExecutionGraph(
        llm=llm,
        environment=environment,
    )

    transitions = TransitionTable()

    transitions.add(
        "awaiting_input",
        "decompose",
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
        graph_name="decomposition_execution",
        current_node="awaiting_input",
        goal=goal,
    )