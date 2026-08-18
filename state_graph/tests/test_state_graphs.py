from __future__ import annotations

import uuid

from state_graph.core.models import GraphState
from state_graph.core.transitions import TransitionTable
from state_graph.core.engine import StateGraphEngine
from state_graph.checkpointing.store import CheckpointStore


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