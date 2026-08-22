from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from state_graph.core.engine import StateGraphEngine
from state_graph.core.models import GraphState
from state_graph.core.transitions import TransitionTable

from .inventory_graph import InventoryGraph


def build_graph3(
    *,
    llm: BaseChatModel,
) -> StateGraphEngine:
    graph = InventoryGraph(llm=llm)

    transitions = TransitionTable()

    transitions.add(
        "awaiting_input",
        "inventory_react",
    )

    transitions.add("inventory_react", "done")

    return StateGraphEngine(
        transitions=transitions,
        nodes=graph.nodes(),
    )


def create_initial_state(run_id: str, goal: str) -> GraphState:
    return GraphState(run_id=run_id, graph_name="inventory", current_node="awaiting_input", goal=goal)
