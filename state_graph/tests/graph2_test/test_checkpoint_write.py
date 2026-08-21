from __future__ import annotations

import sqlite3

from state_graph.core.engine import StateGraphEngine
from state_graph.core.models import GraphState, TransitionResult
from state_graph.core.transitions import TransitionTable
from state_graph.checkpointing.store import CheckpointStore


def make_store(db_path: str) -> CheckpointStore:
    return CheckpointStore(
        connection_factory=lambda: sqlite3.connect(db_path)
    )


def test_checkpoint_written_after_transition(tmp_path):
    db_path = str(tmp_path / "checkpoint.db")

    def awaiting_input(state):
        return TransitionResult(next_node="decompose")

    transitions = TransitionTable()
    transitions.add("awaiting_input", "decompose")

    store = make_store(db_path)

    engine = StateGraphEngine(
        transitions=transitions,
        nodes={"awaiting_input": awaiting_input},
        checkpoint_store=store,
    )

    state = GraphState(
        run_id="run-1",
        graph_name="shipping",
        current_node="awaiting_input",
        goal="Package is missing",
    )

    result = engine.step(state)

    recovered = store.load_latest("run-1")

    assert recovered is not None
    assert recovered.current_node == "decompose"
    assert recovered.transition_count == 1
    assert result.current_node == "decompose"