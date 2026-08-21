from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from state_graph.core.engine import StateGraphEngine
from state_graph.core.models import GraphState, TransitionResult
from state_graph.core.transitions import TransitionTable
from state_graph.graph2.shipping_graph import ShippingGraph


class _FakeCheckpointStore:
    """In-memory checkpoint store standing in for the real (DB-backed) one."""

    def __init__(self) -> None:
        self.saved: list[GraphState] = []

    def save(self, state: GraphState) -> None:
        self.saved.append(state.model_copy())

    def load_latest(self, run_id: str) -> GraphState | None:
        for state in reversed(self.saved):
            if state.run_id == run_id:
                return state
        return None


class _FakeTicketService:
    def create_ticket(self, **kwargs) -> str:
        return "fake-ticket-id"


class _FakeHITLNode:
    """
    Stands in for the DB-backed HITLNode so the test doesn't need a
    real database. Mirrors the same pause()/get_request()/resolve()
    contract.
    """

    def __init__(self) -> None:
        self.requests: dict[str, dict] = {}
        self._next_id = 0

    def pause(self, state: GraphState, *, reason: str) -> TransitionResult:
        self._next_id += 1
        request_id = f"fake-request-{self._next_id}"

        self.requests[request_id] = {
            "request_id": request_id,
            "run_id": state.run_id,
            "graph_name": state.graph_name,
            "reason": reason,
            "state": state.model_dump(mode="json"),
            "decision": None,
            "status": "pending",
        }

        return TransitionResult(
            next_node=state.current_node,
            status="waiting",
            updates={"waiting_request_id": request_id},
        )

    def get_request(self, request_id: str) -> dict | None:
        return self.requests.get(request_id)

    def resolve(self, request_id: str, decision: str) -> None:
        normalized = decision.strip().lower()
        assert normalized in {"approve", "reject"}
        self.requests[request_id]["decision"] = normalized
        self.requests[request_id]["status"] = "resolved"


def _make_engine(graph: ShippingGraph) -> tuple[StateGraphEngine, _FakeCheckpointStore]:
    transitions = TransitionTable()
    transitions.add("awaiting_input", "decompose")
    transitions.add("decompose", "constrained_react")
    transitions.add("constrained_react", "done")

    checkpoint_store = _FakeCheckpointStore()

    engine = StateGraphEngine(
        transitions=transitions,
        nodes=graph.nodes(),
        checkpoint_store=checkpoint_store,
        ticket_service=_FakeTicketService(),
    )

    return engine, checkpoint_store


def _llm_that_escalates() -> MagicMock:
    """LLM double: decomposes fine, then the bound tool-model immediately escalates."""
    llm = MagicMock()

    decompose_response = MagicMock()
    decompose_response.content = "check tracking\nassess claim eligibility"
    llm.invoke.side_effect = [decompose_response]

    react_response = MagicMock()
    react_response.tool_calls = [
        {
            "name": "escalate_to_hitl",
            "args": {"reason": "cannot confidently resolve within allowed tools"},
            "id": "call-1",
        }
    ]
    react_response.content = ""

    bound = MagicMock()
    bound.invoke.side_effect = [react_response]
    llm.bind_tools.return_value = bound

    return llm


def test_hitl_trigger_pauses_and_persists_full_state() -> None:
    """
    A low-confidence (agent-escalated) ReAct result must pause the
    graph — not finish it — and the paused checkpoint must carry the
    full accumulated state (subtasks, react summary), not a partial
    snapshot.
    """
    graph = ShippingGraph(llm=_llm_that_escalates())
    graph.hitl = _FakeHITLNode()

    engine, checkpoint_store = _make_engine(graph)

    state = GraphState(
        run_id="run-1",
        graph_name="shipping",
        current_node="awaiting_input",
        goal="My package never arrived and I want to claim $50 for the lost item.",
    )

    final_state = engine.run(state)

    assert final_state.status == "waiting"
    assert final_state.waiting_request_id is not None
    assert final_state.data.get("hitl_request_id") == final_state.waiting_request_id

    request = graph.hitl.get_request(final_state.waiting_request_id)
    assert request is not None
    assert request["decision"] is None
    assert request["state"]["data"]["subtasks"]
    assert request["state"]["data"]["react"]["confidence"] == 0.0

    latest_checkpoint = checkpoint_store.load_latest("run-1")
    assert latest_checkpoint is not None
    assert latest_checkpoint.status == "waiting"
    assert latest_checkpoint.data["subtasks"]


def test_resume_without_decision_repauses_instead_of_finishing() -> None:
    """
    If resume() is invoked before an admin has actually recorded a
    decision, the graph must re-pause rather than proceed as if
    nothing happened.
    """
    graph = ShippingGraph(llm=_llm_that_escalates())
    graph.hitl = _FakeHITLNode()

    engine, _ = _make_engine(graph)

    state = GraphState(
        run_id="run-2",
        graph_name="shipping",
        current_node="awaiting_input",
        goal="Package lost, claiming $50.",
    )

    paused = engine.run(state)
    assert paused.status == "waiting"

    resumed = engine.resume("run-2")

    assert resumed.status == "waiting"


def test_resume_after_approval_picks_up_the_decision() -> None:
    """
    Once an admin approves the pending request, resuming must reach
    status="done" with the approval recorded — the decision itself
    must be visible in the resumed run's outputs, not just a bare
    "unpaused" state.
    """
    graph = ShippingGraph(llm=_llm_that_escalates())
    graph.hitl = _FakeHITLNode()

    engine, _ = _make_engine(graph)

    state = GraphState(
        run_id="run-3",
        graph_name="shipping",
        current_node="awaiting_input",
        goal="Package lost, claiming $50.",
    )

    paused = engine.run(state)
    assert paused.status == "waiting"

    graph.hitl.resolve(paused.waiting_request_id, "approve")

    resumed = engine.resume("run-3")

    assert resumed.status == "done"
    assert resumed.outputs["hitl_decision"] == "approve"
    assert resumed.data["hitl_decision"] == "approve"
