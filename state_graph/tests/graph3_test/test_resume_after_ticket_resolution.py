from __future__ import annotations

import sqlite3

from state_graph.core.engine import StateGraphEngine
from state_graph.core.models import GraphState, TransitionResult
from state_graph.core.transitions import TransitionTable
from state_graph.checkpointing.store import CheckpointStore
from state_graph.tickets.service import FailureTicketService
from state_graph.graph3.tools import DisputeSyncError


def test_resume_after_ticket_is_resolved(tmp_path):
    db_path = str(tmp_path / "resume.db")

    store = CheckpointStore(connection_factory=lambda: sqlite3.connect(db_path))
    ticket_service = FailureTicketService(connection_factory=lambda: sqlite3.connect(db_path))

    calls = {"count": 0}

    def failing_node(state):
        calls["count"] += 1
        raise DisputeSyncError("Audit_Log write failed after Return_Requests 42 was already set to 'Approved'.")

    transitions = TransitionTable()

    engine = StateGraphEngine(
        transitions=transitions,
        nodes={"dispute_react": failing_node},
        checkpoint_store=store,
        ticket_service=ticket_service,
    )

    state = GraphState(
        run_id="resume-run",
        graph_name="dispute",
        current_node="dispute_react",
        goal="Customer threatens a chargeback on return #42",
    )

    failed = engine.step(state)

    assert failed.status == "failed"
    assert failed.waiting_ticket_id is not None

    ticket_id = failed.waiting_ticket_id
    ticket_service.resolve_ticket(ticket_id, "Audit_Log entry re-synced manually; no duplicate created.")

    def completed_node(state):
        calls["count"] += 1
        return TransitionResult(
            next_node="done",
            status="done",
            updates={"outputs": {"resolution": "Dispute resolved and re-synced."}},
        )

    transitions.add("dispute_react", "done")
    engine.nodes["dispute_react"] = completed_node

    resumed = engine.resume("resume-run")

    assert resumed.status == "done"
    assert resumed.current_node == "done"
    assert resumed.waiting_ticket_id is None
    assert resumed.outputs["resolution"] == "Dispute resolved and re-synced."
    assert calls["count"] == 2