from __future__ import annotations

from state_graph.core.engine import StateGraphEngine
from state_graph.core.transitions import TransitionTable

from .recovery_graph import RecoveryGraph


def build_graph3(
    base_engine: StateGraphEngine,
) -> StateGraphEngine:
    graph = RecoveryGraph(
        engine=base_engine,
    )

    transitions = TransitionTable()

    transitions.add(
        "execute",
        "checkpoint",
    )

    transitions.add(
        "checkpoint",
        "execute",
    )

    transitions.add(
        "execute",
        "classify_failure",
    )

    transitions.add(
        "classify_failure",
        "done",
    )

    transitions.add(
        "classify_failure",
        "ticket",
    )

    transitions.add(
        "ticket",
        "waiting",
    )

    transitions.add(
        "waiting",
        "resume",
    )

    transitions.add(
        "resume",
        "execute",
    )

    return StateGraphEngine(
        transitions=transitions,
        nodes=graph.nodes(),
    )