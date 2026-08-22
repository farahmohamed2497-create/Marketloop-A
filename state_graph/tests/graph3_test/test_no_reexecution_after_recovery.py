from __future__ import annotations

import sqlite3

from state_graph.core.engine import StateGraphEngine
from state_graph.core.models import GraphState, TransitionResult
from state_graph.core.transitions import TransitionTable
from state_graph.checkpointing.store import CheckpointStore


def test_recovery_does_not_reexecute_completed_nodes(tmp_path):
    db_path = str(tmp_path / "no_reexecution.db")

    store = CheckpointStore(connection_factory=lambda: sqlite3.connect(db_path))

    execution_count = {"retention_strategy": 0, "dispute_react": 0}

    def retention_strategy(state):
        execution_count["retention_strategy"] += 1
        return TransitionResult(
            next_node="dispute_react",
            updates={"data": {**state.data, "retention_strategy": {"strategy": "offer partial refund"}}},
        )

    def dispute_react(state):
        execution_count["dispute_react"] += 1
        return TransitionResult(
            next_node="done",
            status="done",
            updates={"outputs": {"resolution": "Dispute resolved."}},
        )

    transitions = TransitionTable()
    transitions.add("retention_strategy", "dispute_react")
    transitions.add("dispute_react", "done")

    engine = StateGraphEngine(
        transitions=transitions,
        nodes={"retention_strategy": retention_strategy, "dispute_react": dispute_react},
        checkpoint_store=store,
    )

    state = GraphState(
        run_id="no-rerun",
        graph_name="dispute",
        current_node="retention_strategy",
        goal="Customer threatens a chargeback on return #42",
    )

    first = engine.step(state)
    assert first.current_node == "dispute_react"
    assert execution_count["retention_strategy"] == 1

    # Simulate process restart.
    new_store = CheckpointStore(connection_factory=lambda: sqlite3.connect(db_path))
    new_engine = StateGraphEngine(
        transitions=transitions,
        nodes={"retention_strategy": retention_strategy, "dispute_react": dispute_react},
        checkpoint_store=new_store,
    )

    recovered = new_engine.recover("no-rerun")
    assert recovered.current_node == "dispute_react"

    final = new_engine.run(recovered)
    assert final.status == "done"

    assert execution_count["retention_strategy"] == 1  # not re-run
    assert execution_count["dispute_react"] == 1