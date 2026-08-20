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