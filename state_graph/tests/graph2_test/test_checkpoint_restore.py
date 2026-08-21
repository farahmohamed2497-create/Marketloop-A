from __future__ import annotations

import sqlite3

from state_graph.core.models import GraphState
from state_graph.checkpointing.store import CheckpointStore
from state_graph.checkpointing.recovery import recover_run


def connection_factory(db_path: str):
    def factory():
        return sqlite3.connect(db_path)

    return factory


def test_recover_latest_checkpoint(tmp_path):
    db_path = str(tmp_path / "graph2.db")

    factory = connection_factory(db_path)

    first_store = CheckpointStore(
        connection_factory=factory
    )

    state = GraphState(
        run_id="shipping-recovery-001",
        graph_name="shipping",
        current_node="constrained_react",
        goal="Shipment is missing",
        status="running",
        transition_count=2,
        data={
            "subtasks": [
                "Check tracking",
                "Investigate missing shipment",
            ]
        },
    )

    first_store.save(state)

    # Simulate the original process ending.
    del first_store

    # New store represents a new process.
    restarted_store = CheckpointStore(
        connection_factory=factory
    )

    recovered = recover_run(
        run_id="shipping-recovery-001",
        store=restarted_store,
    )

    assert recovered is not None
    assert recovered.run_id == "shipping-recovery-001"
    assert recovered.current_node == "constrained_react"
    assert recovered.transition_count == 2

    assert recovered.data["subtasks"] == [
        "Check tracking",
        "Investigate missing shipment",
    ]