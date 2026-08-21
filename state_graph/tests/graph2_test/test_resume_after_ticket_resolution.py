from __future__ import annotations

import sqlite3

from state_graph.core.engine import StateGraphEngine
from state_graph.core.models import GraphState, TransitionResult
from state_graph.core.transitions import TransitionTable
from state_graph.checkpointing.store import CheckpointStore
from state_graph.tickets.service import FailureTicketService


def test_resume_after_ticket_is_resolved(tmp_path):
    db_path = str(tmp_path / "resume.db")

    store = CheckpointStore(
        connection_factory=lambda: sqlite3.connect(db_path)
    )

    ticket_service = FailureTicketService(
        connection_factory=lambda: sqlite3.connect(db_path)
    )

    calls = {"count": 0}

    def failing_node(state):
        calls["count"] += 1
        raise RuntimeError("temporary carrier investigation failure")

    transitions = TransitionTable()

    engine = StateGraphEngine(
        transitions=transitions,
        nodes={"investigate": failing_node},
        checkpoint_store=store,
        ticket_service=ticket_service,
    )

    state = GraphState(
        run_id="resume-run",
        graph_name="shipping",
        current_node="investigate",
        goal="Investigate missing package",
    )

    failed = engine.step(state)

    assert failed.status == "failed"
    assert failed.waiting_ticket_id is not None

    ticket_id = failed.waiting_ticket_id

    ticket_service.resolve_ticket(
        ticket_id,
        "Carrier investigation completed.",
    )

    def completed_node(state):
        calls["count"] += 1
        return TransitionResult(
            next_node="done",
            status="done",
            updates={
                "outputs": {
                    "resolution": "Carrier investigation completed."
                }
            },
        )

    transitions.add("investigate", "done")

    engine.nodes["investigate"] = completed_node

    resumed = engine.resume("resume-run")

    assert resumed.status == "done"
    assert resumed.current_node == "done"
    assert resumed.waiting_ticket_id is None
    assert resumed.outputs["resolution"] == (
        "Carrier investigation completed."
    )

    assert calls["count"] == 2