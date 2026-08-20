from __future__ import annotations

import random
import uuid
from unittest.mock import MagicMock

from planning_lab.algorithms.environment import Environment
from state_graph.checkpointing.store import CheckpointStore
from state_graph.core.engine import StateGraphEngine
from state_graph.core.models import GraphState
from state_graph.core.transitions import TransitionTable
from state_graph.graph1.refund_graph import RefundGraph


from state_graph.hitl.policy import requires_human_intervention
from state_graph.hitl.node import HITLNode

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


def test_refund_lats_node_stores_result():
    mock_llm = MagicMock()

    graph = RefundGraph(
        llm=mock_llm,
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

    assert "output" in lats_data
    assert "best_score" in lats_data
    assert "iterations" in lats_data

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