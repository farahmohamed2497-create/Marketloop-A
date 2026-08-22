from __future__ import annotations

import sqlite3

from state_graph.core.engine import StateGraphEngine
from state_graph.core.models import GraphState
from state_graph.core.transitions import TransitionTable
from state_graph.checkpointing.store import CheckpointStore
from state_graph.tickets.service import FailureTicketService
from state_graph.graph3.tools import DisputeSyncError


def test_unplanned_failure_creates_recovery_ticket(tmp_path):
    db_path = str(tmp_path / "tickets.db")

    def broken_node(state):
        raise DisputeSyncError("Audit_Log write failed after Return_Requests 42 was already set to 'Approved'.")

    store = CheckpointStore(connection_factory=lambda: sqlite3.connect(db_path))
    ticket_service = FailureTicketService(connection_factory=lambda: sqlite3.connect(db_path))

    engine = StateGraphEngine(
        transitions=TransitionTable(),
        nodes={"dispute_react": broken_node},
        checkpoint_store=store,
        ticket_service=ticket_service,
    )

    state = GraphState(
        run_id="ticket-run",
        graph_name="dispute",
        current_node="dispute_react",
        goal="Customer threatens a chargeback on return #42",
    )

    result = engine.step(state)

    assert result.status == "failed"
    assert result.waiting_ticket_id is not None

    ticket = ticket_service.get_ticket(result.waiting_ticket_id)

    assert ticket is not None
    assert ticket["run_id"] == "ticket-run"
    assert ticket["graph_name"] == "dispute"
    assert ticket["node_name"] == "dispute_react"
    assert ticket["status"] == "open"
    assert "Audit_Log" in ticket["error"]