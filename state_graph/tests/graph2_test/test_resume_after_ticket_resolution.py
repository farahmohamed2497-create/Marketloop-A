from __future__ import annotations

import sqlite3

from state_graph.core.models import GraphState
from state_graph.checkpointing.store import CheckpointStore
from state_graph.tickets.service import FailureTicketService
from state_graph.checkpointing.recovery import recover_run


def connection_factory(db_path: str):
    def factory():
        return sqlite3.connect(db_path)

    return factory


def test_resume_after_ticket_resolution(tmp_path):
    db_path = str(tmp_path / "graph2.db")

    factory = connection_factory(db_path)

    checkpoint_store = CheckpointStore(
        connection_factory=factory
    )

    ticket_service = FailureTicketService(
        connection_factory=factory
    )

    failed_state = GraphState(
        run_id="shipping-resume-001",
        graph_name="shipping",
        current_node="constrained_react",
        goal="Shipment was not delivered",
        status="failed",
        transition_count=2,
        data={
            "subtasks": [
                "Check tracking",
                "Investigate carrier response",
            ],
            "last_completed_step": "check_tracking",
        },
    )

    checkpoint_store.save(failed_state)

    ticket_id = ticket_service.create_ticket(
        run_id=failed_state.run_id,
        graph_name=failed_state.graph_name,
        node_name=failed_state.current_node,
        error="Carrier API returned contradictory data.",
        state=failed_state.model_dump(mode="json"),
    )

    ticket = ticket_service.get_ticket(ticket_id)

    assert ticket is not None
    assert ticket["status"] == "open"

    ticket_service.begin_investigation(ticket_id)

    ticket = ticket_service.get_ticket(ticket_id)

    assert ticket["status"] == "investigating"

    ticket_service.resolve_ticket(
        ticket_id,
        resolution="Carrier data corrected.",
    )

    ticket = ticket_service.get_ticket(ticket_id)

    assert ticket["status"] == "resolved"

    # Simulate a fresh process after ticket resolution.
    restarted_store = CheckpointStore(
        connection_factory=factory
    )

    recovered = recover_run(
        run_id=failed_state.run_id,
        store=restarted_store,
    )

    assert recovered is not None

    # Must resume from the checkpointed node.
    assert recovered.current_node == "constrained_react"

    # Must preserve previous progress.
    assert recovered.transition_count == 2

    assert (
        recovered.data["last_completed_step"]
        == "check_tracking"
    )