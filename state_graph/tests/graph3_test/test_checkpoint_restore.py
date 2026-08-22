from __future__ import annotations

import sqlite3

from state_graph.core.engine import StateGraphEngine
from state_graph.core.models import GraphState
from state_graph.core.transitions import TransitionTable
from state_graph.checkpointing.store import CheckpointStore


def make_store(db_path: str) -> CheckpointStore:
    return CheckpointStore(connection_factory=lambda: sqlite3.connect(db_path))


def test_checkpoint_restore_after_process_restart(tmp_path):
    db_path = str(tmp_path / "checkpoint.db")

    store_1 = make_store(db_path)

    state = GraphState(
        run_id="run-restart",
        graph_name="dispute",
        current_node="dispute_react",
        goal="Customer threatens legal action over return #42",
        transition_count=2,
        data={"retention_strategy": {"strategy": "offer partial refund", "score": 0.8}},
    )

    engine_1 = StateGraphEngine(transitions=TransitionTable(), nodes={}, checkpoint_store=store_1)
    engine_1.initialize(state)

    # Simulate a new process reopening the same database.
    store_2 = make_store(db_path)
    engine_2 = StateGraphEngine(transitions=TransitionTable(), nodes={}, checkpoint_store=store_2)

    recovered = engine_2.recover("run-restart")

    assert recovered.run_id == "run-restart"
    assert recovered.current_node == "dispute_react"
    assert recovered.transition_count == 2
    assert recovered.data["retention_strategy"]["strategy"]