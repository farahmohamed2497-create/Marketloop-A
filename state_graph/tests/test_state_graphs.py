from __future__ import annotations

import random
import sqlite3
import subprocess
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock

from pydantic import BaseModel
import pytest

from planning_lab.algorithms.environment import Environment
from state_graph.checkpointing.store import CheckpointStore
from state_graph.core.engine import StateGraphEngine
from state_graph.core.models import GraphState, TransitionResult
from state_graph.core.transitions import TransitionTable
from state_graph.graph1.refund_graph import RefundGraph
from state_graph.tickets.service import FailureTicketService

from types import SimpleNamespace

from state_graph.hitl.node import HITLNode
from state_graph.hitl.policy import requires_human_intervention


def test_checkpoint_round_trip():
    store = CheckpointStore()

    state = GraphState(
        run_id=str(uuid.uuid4()),
        graph_name="test",
        current_node="start",
        status="running",
        goal="Test graph persistence",
        data={"x": 1},
        outputs={"result": "ok"},
    )

    store.save(state)

    recovered = store.load_latest(
        state.run_id
    )

    assert recovered is not None
    assert recovered.run_id == state.run_id
    assert recovered.current_node == "start"
    assert recovered.data["x"] == 1


def test_transition_table():
    transitions = TransitionTable()

    transitions.add(
        "start",
        "middle",
    )

    transitions.add(
        "middle",
        "done",
    )

    assert transitions.allowed(
        "start",
        "middle",
    )

    assert not transitions.allowed(
        "start",
        "done",
    )


def test_engine_persists_state():
    transitions = TransitionTable()

    transitions.add(
        "start",
        "done",
    )

    def start(state):
        from state_graph.core.models import TransitionResult

        return TransitionResult(
            next_node="done",
            status="done",
        )

    engine = StateGraphEngine(
        transitions=transitions,
        nodes={
            "start": start,
        },
    )

    state = GraphState(
        run_id=str(uuid.uuid4()),
        graph_name="test",
        current_node="start",
        goal="Test persistent execution",
    )

    result = engine.run(state)

    assert result.status == "done"

    recovered = engine.recover(
        state.run_id
    )

    assert recovered.status == "done"
    assert recovered.current_node == "done"


def test_graph1_checkpoints_initial_and_each_transition():
    """Graph 1 keeps a recoverable snapshot at every meaningful step."""

    class RecordingCheckpointStore:
        def __init__(self):
            self.snapshots: list[GraphState] = []

        def save(self, state: GraphState) -> None:
            self.snapshots.append(state.model_copy(deep=True))

    transitions = TransitionTable()
    transitions.add("awaiting_input", "policy_check")
    transitions.add("policy_check", "done")

    def awaiting_input(_state: GraphState) -> TransitionResult:
        return TransitionResult(
            next_node="policy_check",
            updates={"data": {"request_id": "return-42"}},
        )

    def policy_check(_state: GraphState) -> TransitionResult:
        return TransitionResult(next_node="done", status="done")

    store = RecordingCheckpointStore()
    engine = StateGraphEngine(
        transitions=transitions,
        nodes={
            "awaiting_input": awaiting_input,
            "policy_check": policy_check,
        },
        checkpoint_store=store,
    )
    state = GraphState(
        run_id=str(uuid.uuid4()),
        graph_name="refund",
        current_node="awaiting_input",
        goal="Process a damaged-item refund request.",
    )

    result = engine.run(state)

    assert result.status == "done"
    assert [snapshot.current_node for snapshot in store.snapshots] == [
        "awaiting_input",
        "policy_check",
        "done",
    ]
    assert [snapshot.transition_count for snapshot in store.snapshots] == [0, 1, 2]
    assert store.snapshots[1].data == {"request_id": "return-42"}


def test_graph1_restores_latest_checkpoint_after_store_restart(tmp_path):
    """A new checkpoint-store instance resumes Graph 1 from durable state."""

    database_path = tmp_path / "graph1-checkpoints.db"

    def connect():
        return sqlite3.connect(database_path)

    original_store = CheckpointStore(connection_factory=connect)
    initial = GraphState(
        run_id=str(uuid.uuid4()),
        graph_name="refund",
        current_node="awaiting_inspection",
        status="waiting",
        goal="Refund a damaged item after warehouse inspection.",
        data={"return_id": "RET-42", "photos_received": True},
        transition_count=3,
    )
    original_store.save(initial)

    restarted_store = CheckpointStore(connection_factory=connect)
    recovered = restarted_store.load_latest(initial.run_id)

    assert recovered == initial


def test_graph1_classifies_tool_and_schema_failures():
    class RecordingCheckpointStore:
        def save(self, _state: GraphState) -> None:
            pass

    class RecordingTicketService:
        def __init__(self):
            self.errors: list[str] = []

        def create_ticket(self, **kwargs) -> str:
            self.errors.append(kwargs["error"])
            return "ticket-42"

    class ReturnPayload(BaseModel):
        order_id: int

    def tool_node(_state: GraphState) -> TransitionResult:
        raise TimeoutError("carrier tool timed out")

    def schema_node(_state: GraphState) -> TransitionResult:
        ReturnPayload.model_validate({"order_id": "not-an-integer"})
        raise AssertionError("ValidationError should have been raised")

    transitions = TransitionTable()
    ticket_service = RecordingTicketService()
    engine = StateGraphEngine(
        transitions=transitions,
        nodes={"tool": tool_node, "schema": schema_node},
        checkpoint_store=RecordingCheckpointStore(),
        ticket_service=ticket_service,
    )

    tool_failure = engine.step(
        GraphState(
            run_id=str(uuid.uuid4()),
            graph_name="refund",
            current_node="tool",
        )
    )
    schema_failure = engine.step(
        GraphState(
            run_id=str(uuid.uuid4()),
            graph_name="refund",
            current_node="schema",
        )
    )

    assert tool_failure.data["failure"]["kind"] == "tool_error"
    assert schema_failure.data["failure"]["kind"] == "schema_validation_error"
    assert tool_failure.waiting_ticket_id == "ticket-42"
    assert len(ticket_service.errors) == 2


def test_graph1_failure_ticket_persists_failure_state(tmp_path):
    database_path = tmp_path / "graph1-tickets.db"

    def connect():
        return sqlite3.connect(database_path)

    service = FailureTicketService(connection_factory=connect)
    ticket_id = service.create_ticket(
        run_id="refund-run-42",
        graph_name="refund",
        node_name="inspection",
        error="warehouse tool returned malformed payload",
        state={"current_node": "inspection", "return_id": "RET-42"},
    )

    ticket = service.get_ticket(ticket_id)

    assert ticket is not None
    assert ticket["status"] == "open"
    assert ticket["state"]["return_id"] == "RET-42"
    assert ticket["node_name"] == "inspection"

    service.begin_investigation(ticket_id)
    assert service.get_ticket(ticket_id)["status"] == "investigating"

    service.resolve_ticket(ticket_id)
    assert service.get_ticket(ticket_id)["status"] == "resolved"


def test_graph1_ticket_captures_the_classified_failed_checkpoint():
    class RecordingCheckpointStore:
        def save(self, _state: GraphState) -> None:
            pass

    class RecordingTicketService:
        def __init__(self):
            self.state = None

        def create_ticket(self, **kwargs) -> str:
            self.state = kwargs["state"]
            return "ticket-42"

    def inspection(_state: GraphState) -> TransitionResult:
        raise TimeoutError("inspection tool timed out")

    ticket_service = RecordingTicketService()
    engine = StateGraphEngine(
        transitions=TransitionTable(),
        nodes={"inspection": inspection},
        checkpoint_store=RecordingCheckpointStore(),
        ticket_service=ticket_service,
    )

    failed = engine.step(
        GraphState(
            run_id=str(uuid.uuid4()),
            graph_name="refund",
            current_node="inspection",
        )
    )

    assert failed.waiting_ticket_id == "ticket-42"
    assert ticket_service.state["status"] == "failed"
    assert ticket_service.state["data"]["failure"]["kind"] == "tool_error"


def test_graph1_resumes_only_after_failure_ticket_is_resolved(tmp_path):
    database_path = tmp_path / "graph1-recovery.db"

    def connect():
        return sqlite3.connect(database_path)

    repaired = False

    def inspect_return(_state: GraphState) -> TransitionResult:
        if not repaired:
            raise TimeoutError("warehouse inspection tool is unavailable")
        return TransitionResult(next_node="done", status="done")

    engine = StateGraphEngine(
        transitions=TransitionTable(),
        nodes={"inspection": inspect_return},
        checkpoint_store=CheckpointStore(connection_factory=connect),
        ticket_service=FailureTicketService(connection_factory=connect),
    )
    engine.transitions.add("inspection", "done")
    initial = GraphState(
        run_id=str(uuid.uuid4()),
        graph_name="refund",
        current_node="inspection",
        goal="Inspect return RET-42 before refunding it.",
    )

    failed = engine.run(initial)

    assert failed.status == "failed"
    assert failed.waiting_ticket_id is not None
    with pytest.raises(ValueError, match="must be resolved"):
        engine.resume(initial.run_id)

    engine.ticket_service.resolve_ticket(failed.waiting_ticket_id)
    repaired = True
    resumed = engine.resume(initial.run_id)

    assert resumed.status == "done"
    assert resumed.current_node == "done"
    assert resumed.transition_count == 1


def test_graph1_recovers_after_process_is_killed(tmp_path):
    """A restarted process continues after, not before, the last checkpoint."""

    database_path = tmp_path / "graph1-crash-recovery.db"
    run_id = str(uuid.uuid4())
    project_root = Path(__file__).resolve().parents[2]
    child_program = f"""
import os
import sqlite3
from state_graph.checkpointing.store import CheckpointStore
from state_graph.core.engine import StateGraphEngine
from state_graph.core.models import GraphState, TransitionResult
from state_graph.core.transitions import TransitionTable

def connect():
    return sqlite3.connect({str(database_path)!r})

def collect_return(state):
    return TransitionResult(
        next_node='awaiting_inspection',
        updates={{'data': {{'return_id': 'RET-42'}}}},
    )

def await_inspection(_state):
    os._exit(86)

transitions = TransitionTable()
transitions.add('collect_return', 'awaiting_inspection')
engine = StateGraphEngine(
    transitions=transitions,
    nodes={{
        'collect_return': collect_return,
        'awaiting_inspection': await_inspection,
    }},
    checkpoint_store=CheckpointStore(connection_factory=connect),
    ticket_service=object(),
)
engine.run(GraphState(
    run_id={run_id!r},
    graph_name='refund',
    current_node='collect_return',
    goal='Collect and inspect a damaged-item return.',
))
"""

    crashed = subprocess.run(
        [sys.executable, "-c", child_program],
        cwd=project_root,
        check=False,
    )

    assert crashed.returncode == 86

    def connect():
        return sqlite3.connect(database_path)

    def collect_return_should_not_run(_state: GraphState) -> TransitionResult:
        raise AssertionError("Completed node was executed after restart")

    def await_inspection(_state: GraphState) -> TransitionResult:
        return TransitionResult(next_node="done", status="done")

    transitions = TransitionTable()
    transitions.add("collect_return", "awaiting_inspection")
    transitions.add("awaiting_inspection", "done")
    restarted_engine = StateGraphEngine(
        transitions=transitions,
        nodes={
            "collect_return": collect_return_should_not_run,
            "awaiting_inspection": await_inspection,
        },
        checkpoint_store=CheckpointStore(connection_factory=connect),
        ticket_service=object(),
    )

    checkpoint = restarted_engine.recover(run_id)
    resumed = restarted_engine.run(checkpoint)

    assert checkpoint.current_node == "awaiting_inspection"
    assert checkpoint.data == {"return_id": "RET-42"}
    assert resumed.status == "done"
    assert resumed.transition_count == 2


def test_refund_lats_node_stores_result(monkeypatch):
    fake_lats_result = SimpleNamespace(
        success=True,
        output="Full refund is appropriate.",
        best_score=0.90,
        iterations=2,
    )

    monkeypatch.setattr(
        "state_graph.graph1.refund_graph.lats",
        lambda **kwargs: fake_lats_result,
    )

    graph = RefundGraph(
        llm=MagicMock(),
        environment=Environment(
            success_threshold=0.0,
            rng=random.Random(42),
        ),
    )

    state = GraphState(
        run_id=str(uuid.uuid4()),
        graph_name="refund",
        current_node="lats",
        goal="Customer requests a refund for order ORD-123.",
    )

    result = graph.lats_node(state)

    assert result.next_node == "evaluate_refund"

    assert "lats" in result.updates["data"]

    lats_data = result.updates["data"]["lats"]

    assert lats_data["output"] == "Full refund is appropriate."
    assert lats_data["best_score"] == 0.90
    assert lats_data["iterations"] == 2
    #test human_intervention
  


def test_hitl_triggers_on_low_confidence():
    assert requires_human_intervention(
        score=0.69,
    )


def test_hitl_does_not_trigger_on_sufficient_confidence():
    assert not requires_human_intervention(
        score=0.70,
    )


def test_hitl_triggers_on_high_refund_amount():
    assert requires_human_intervention(
        refund_amount=500.01,
    )


def test_hitl_does_not_trigger_below_refund_threshold():
    assert not requires_human_intervention(
        refund_amount=500.0,
    )


def test_hitl_triggers_on_policy_violation():
    assert requires_human_intervention(
        policy_violation=True,
    )


def test_hitl_does_not_trigger_when_policy_is_compliant():
    assert not requires_human_intervention(
        policy_violation=False,
    )



    #test db
def test_hitl_node_pauses_and_persists_full_state():
    hitl = HITLNode()

    state = GraphState(
        run_id=str(uuid.uuid4()),
        graph_name="refund",
        current_node="hitl",
        status="running",
        goal="Customer requests a refund for order ORD-123.",
        data={
            "refund_amount": 750.0,
            "lats_score": 0.62,
            "proposed_action": "partial_refund",
        },
        outputs={
            "customer": "test-customer",
        },
    )

    result = hitl.pause(
        state,
        reason="Refund amount exceeds the approval threshold.",
    )

    assert result.next_node == "hitl"
    assert result.status == "waiting"

    request_id = result.updates["waiting_request_id"]

    assert request_id

    request = hitl.get_request(request_id)

    assert request is not None
    assert request["run_id"] == state.run_id
    assert request["graph_name"] == state.graph_name
    assert request["status"] == "pending"
    assert request["decision"] is None

    persisted_state = request["state"]

    assert persisted_state["status"] == "waiting"
    assert persisted_state["waiting_request_id"] == request_id
    assert persisted_state["goal"] == state.goal
    assert persisted_state["data"]["refund_amount"] == 750.0
    assert persisted_state["data"]["lats_score"] == 0.62
    assert (
        persisted_state["data"]["proposed_action"]
        == "partial_refund"
    )
    assert persisted_state["outputs"]["customer"] == "test-customer"


    #test simulate HITL trigger and verify pause behavior

def test_refund_graph_triggers_hitl_on_low_confidence(
    monkeypatch,
):
     fake_lats_result = SimpleNamespace(
        success=False,
        output="Partial refund is the safest option.",
        best_score=0.55,
        iterations=2,
    )

     monkeypatch.setattr(
        "state_graph.graph1.refund_graph.lats",
        lambda **kwargs: fake_lats_result,
    )

     graph = RefundGraph(
        llm=MagicMock(),
        environment=Environment(
            success_threshold=0.0,
            rng=random.Random(42),
        ),
    )

     state = GraphState(
        run_id=str(uuid.uuid4()),
        graph_name="refund",
        current_node="lats",
        goal="Customer requests a refund for order ORD-123.",
        data={
            "refund_amount": 100.0,
        },
    )

     result = graph.lats_node(state)

     assert result.status == "waiting"

     request_id = result.updates["waiting_request_id"]

     assert request_id

     request = graph.hitl.get_request(
        request_id
    )

     assert request is not None
     assert request["status"] == "pending"
     assert request["run_id"] == state.run_id

     persisted_state = request["state"]

     assert persisted_state["status"] == "waiting"
     assert (
        persisted_state["data"]["lats"]["best_score"]
        == 0.55
    )
