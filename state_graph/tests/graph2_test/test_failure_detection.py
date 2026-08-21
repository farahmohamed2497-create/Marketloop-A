from __future__ import annotations

import sqlite3

from pydantic import ValidationError

from state_graph.core.engine import StateGraphEngine
from state_graph.core.models import GraphState
from state_graph.core.transitions import TransitionTable
from state_graph.checkpointing.store import CheckpointStore
from state_graph.tickets.service import FailureTicketService


def make_store(db_path: str) -> CheckpointStore:
    return CheckpointStore(
        connection_factory=lambda: sqlite3.connect(db_path)
    )


def make_ticket_service(db_path: str) -> FailureTicketService:
    return FailureTicketService(
        connection_factory=lambda: sqlite3.connect(db_path)
    )


def make_state():
    return GraphState(
        run_id="failure-run",
        graph_name="shipping",
        current_node="check_tracking",
        goal="Package is missing",
    )


def test_tool_error_is_classified_and_persisted(tmp_path):
    db_path = str(tmp_path / "failure.db")

    def failing_tool(state):
        raise ConnectionError("tracking API failed")

    store = make_store(db_path)
    tickets = make_ticket_service(db_path)

    engine = StateGraphEngine(
        transitions=TransitionTable(),
        nodes={"check_tracking": failing_tool},
        checkpoint_store=store,
        ticket_service=tickets,
    )

    result = engine.step(make_state())

    assert result.status == "failed"
    assert result.data["failure"]["kind"] == "tool_error"
    assert result.data["failure"]["node"] == "check_tracking"
    assert result.waiting_ticket_id is not None


def test_schema_validation_failure_is_classified(tmp_path):
    db_path = str(tmp_path / "schema_failure.db")

    def failing_tool(state):
        raise ValidationError.from_exception_data(
            "GraphState",
            [
                {
                    "type": "missing",
                    "loc": ("tracking_number",),
                    "input": {},
                }
            ],
        )

    store = make_store(db_path)
    tickets = make_ticket_service(db_path)

    engine = StateGraphEngine(
        transitions=TransitionTable(),
        nodes={"check_tracking": failing_tool},
        checkpoint_store=store,
        ticket_service=tickets,
    )

    result = engine.step(make_state())

    assert result.status == "failed"
    assert result.data["failure"]["kind"] == "schema_validation_error"
    assert result.waiting_ticket_id is not None