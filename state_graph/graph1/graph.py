from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from planning_lab.algorithms.environment import Environment

from state_graph.core.engine import StateGraphEngine
from state_graph.core.models import GraphState
from state_graph.core.transitions import TransitionTable

from .decomposition_execution import (
    DecompositionExecutionGraph,
)


def build_graph1(
    *,
    llm: BaseChatModel,
    environment: Environment | None = None,
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

    transitions.add(
        "decompose",
        "validate_plan",
    )

    transitions.add(
        "validate_plan",
        "execute_parallel_tasks",
    )

    transitions.add(
        "execute_parallel_tasks",
        "ground_result",
    )

    transitions.add(
        "ground_result",
        "synthesis",
    )

    transitions.add(
        "ground_result",
        "retry_refine",
    )

    transitions.add(
        "retry_refine",
        "execute_parallel_tasks",
    )

    transitions.add(
        "synthesis",
        "done",
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