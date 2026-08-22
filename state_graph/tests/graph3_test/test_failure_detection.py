from __future__ import annotations

import sqlite3

from state_graph.core.engine import StateGraphEngine
from state_graph.core.models import GraphState
from state_graph.core.transitions import TransitionTable
from state_graph.checkpointing.store import CheckpointStore
from state_graph.tickets.service import FailureTicketService
from state_graph.graph3.tools import DisputeSyncError


def make_store(db_path: str) -> CheckpointStore:
    return CheckpointStore(connection_factory=lambda: sqlite3.connect(db_path))


def make_ticket_service(db_path: str) -> FailureTicketService:
    return FailureTicketService(connection_factory=lambda: sqlite3.connect(db_path))


def make_state():
    return GraphState(
        run_id="failure-run",
        graph_name="dispute",
        current_node="dispute_react",
        goal="Customer threatens a chargeback on return #42",
    )


def test_dispute_sync_error_is_classified_as_tool_error(tmp_path):
    db_path = str(tmp_path / "failure.db")

    def failing_node(state):
        raise DisputeSyncError("Audit_Log write failed after Return_Requests 42 was already set to 'Approved'.")

    engine = StateGraphEngine(
        transitions=TransitionTable(),
        nodes={"dispute_react": failing_node},
        checkpoint_store=make_store(db_path),
        ticket_service=make_ticket_service(db_path),
    )

    result = engine.step(make_state())

    assert result.status == "failed"
    assert result.data["failure"]["kind"] == "tool_error"
    assert result.data["failure"]["node"] == "dispute_react"
    assert result.waiting_ticket_id is not None


def test_disallowed_tool_call_is_classified_as_unplanned(tmp_path):
    db_path = str(tmp_path / "schema_failure.db")

    from planning_lab.algorithms.react import ToolNotAllowedError

    def failing_node(state):
        raise ToolNotAllowedError("Model attempted to call disallowed tool: 'apply_refund_directly'")

    engine = StateGraphEngine(
        transitions=TransitionTable(),
        nodes={"dispute_react": failing_node},
        checkpoint_store=make_store(db_path),
        ticket_service=make_ticket_service(db_path),
    )

    result = engine.step(make_state())

    assert result.status == "failed"
    assert result.data["failure"]["kind"] == "unplanned_error"
    assert result.waiting_ticket_id is not None