from __future__ import annotations

from unittest.mock import MagicMock

from state_graph.core.engine import StateGraphEngine
from state_graph.core.models import GraphState, TransitionResult
from state_graph.core.transitions import TransitionTable
from state_graph.graph3.dispute_graph import DisputeGraph


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
    """Mirrors HITLNode's pause()/get_request()/resolve() contract without a DB."""

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


class _FakeAdminTaskQueue:
    """Stands in for DatabaseAdminTaskQueue so no real Admin_Task_Queue write happens."""

    def __init__(self) -> None:
        self.enqueued: list[dict] = []

    def enqueue_hitl(self, *, request_id, run_id, graph_name, state) -> None:
        self.enqueued.append(
            {"request_id": request_id, "run_id": run_id, "graph_name": graph_name, "state": state}
        )


def _make_engine(graph: DisputeGraph) -> tuple[StateGraphEngine, _FakeCheckpointStore]:
    transitions = TransitionTable()
    transitions.add("awaiting_input", "retention_strategy")
    transitions.add("retention_strategy", "dispute_react")
    transitions.add("dispute_react", "awaiting_customer_response")
    transitions.add("dispute_react", "done")
    transitions.add("awaiting_customer_response", "dispute_react")

    checkpoint_store = _FakeCheckpointStore()

    engine = StateGraphEngine(
        transitions=transitions,
        nodes=graph.nodes(),
        checkpoint_store=checkpoint_store,
        ticket_service=_FakeTicketService(),
    )

    return engine, checkpoint_store


def _llm_that_resolves_calmly() -> MagicMock:
    """LLM double: ReAct proposes an offer and stops without self-escalating.

    Used to prove the HITL pause fires from the *policy* (legal_threat=True
    in state.data), not just from the agent asking to escalate.
    """
    llm = MagicMock()

    react_response = MagicMock()
    react_response.tool_calls = []
    react_response.content = "Proposed a 10% discount on the next order as a retention offer."

    bound = MagicMock()
    bound.invoke.side_effect = [react_response]
    llm.bind_tools.return_value = bound

    return llm


def _llm_that_escalates() -> MagicMock:
    """LLM double: the bound tool-model immediately calls escalate_to_hitl."""
    llm = MagicMock()

    react_response = MagicMock()
    react_response.tool_calls = [
        {
            "name": "escalate_to_hitl",
            "args": {"reason": "customer confidence too low to propose an offer safely"},
            "id": "call-1",
        }
    ]
    react_response.content = ""

    bound = MagicMock()
    bound.invoke.side_effect = [react_response]
    llm.bind_tools.return_value = bound

    return llm


def _dispute_state(run_id: str, **data_overrides) -> GraphState:
    data = {"return_id": 1, "legal_threat": False, "retention_offer_value": 0}
    data.update(data_overrides)
    return GraphState(
        run_id=run_id,
        graph_name="dispute",
        current_node="dispute_react",
        goal="Customer threatens a chargeback on return #1 unless offered a discount.",
        data=data,
    )


def test_legal_threat_pauses_and_persists_full_state_even_when_react_succeeds() -> None:
    """
    A confident, successful ReAct result must still pause the graph when
    the customer raised a legal threat -- the HITL policy, not the
    agent's own confidence, is what decides here. The paused checkpoint
    must carry the full accumulated state (the react summary), not a
    partial snapshot.
    """
    graph = DisputeGraph(llm=_llm_that_resolves_calmly(), task_queue=_FakeAdminTaskQueue())
    graph.hitl = _FakeHITLNode()

    engine, checkpoint_store = _make_engine(graph)

    state = _dispute_state("run-1", legal_threat=True, retention_offer_value=100)

    final_state = engine.run(state)

    assert final_state.status == "waiting"
    assert final_state.waiting_request_id is not None
    assert final_state.data.get("hitl_request_id") == final_state.waiting_request_id

    request = graph.hitl.get_request(final_state.waiting_request_id)
    assert request is not None
    assert request["decision"] is None
    assert request["state"]["data"]["react"]["confidence"] == 1.0
    assert "discount" in request["state"]["data"]["react"]["output"]

    latest_checkpoint = checkpoint_store.load_latest("run-1")
    assert latest_checkpoint is not None
    assert latest_checkpoint.status == "waiting"
    assert latest_checkpoint.data["react"]["success"] is True

    assert len(graph.task_queue.enqueued) == 1
    assert graph.task_queue.enqueued[0]["run_id"] == "run-1"


def test_agent_self_escalation_also_pauses_for_compliance() -> None:
    """The agent's own escalate_to_hitl call must pause the graph too,
    independently of the legal_threat / offer-value policy checks."""
    graph = DisputeGraph(llm=_llm_that_escalates(), task_queue=_FakeAdminTaskQueue())
    graph.hitl = _FakeHITLNode()

    engine, _ = _make_engine(graph)

    state = _dispute_state("run-2")  # legal_threat=False, offer below threshold

    final_state = engine.run(state)

    assert final_state.status == "waiting"
    assert final_state.waiting_request_id is not None
    assert final_state.data["react"]["confidence"] == 0.0


def test_resume_without_decision_repauses_instead_of_finishing() -> None:
    """If resume() runs before an admin has recorded a decision, the
    graph must re-pause rather than proceed as if nothing happened."""
    graph = DisputeGraph(llm=_llm_that_escalates(), task_queue=_FakeAdminTaskQueue())
    graph.hitl = _FakeHITLNode()

    engine, _ = _make_engine(graph)

    state = _dispute_state("run-3")

    paused = engine.run(state)
    assert paused.status == "waiting"

    resumed = engine.resume("run-3")

    assert resumed.status == "waiting"


def test_resume_after_approval_picks_up_the_decision_and_syncs(monkeypatch) -> None:
    """
    Once an admin approves the pending request, resuming must reach
    status="done" with the approval recorded, and it must call
    sync_dispute_resolution exactly once with the admin's decision --
    not re-run the ReAct loop, and not proceed as a bare "unpaused"
    state with no resolution attached.
    """
    sync_calls: list[dict] = []

    def fake_sync_dispute_resolution(**kwargs):
        sync_calls.append(kwargs)
        return {"return_id": kwargs["return_id"], "decision": kwargs["decision"], "synced": True}

    monkeypatch.setattr(
        "state_graph.graph3.dispute_graph.sync_dispute_resolution",
        fake_sync_dispute_resolution,
    )

    graph = DisputeGraph(llm=_llm_that_escalates(), task_queue=_FakeAdminTaskQueue())
    graph.hitl = _FakeHITLNode()

    engine, _ = _make_engine(graph)

    state = _dispute_state("run-4")

    paused = engine.run(state)
    assert paused.status == "waiting"

    graph.hitl.resolve(paused.waiting_request_id, "approve")

    resumed = engine.resume("run-4")

    assert resumed.status == "done"
    assert resumed.outputs["hitl_decision"] == "approve"
    assert resumed.data["hitl_decision"] == "approve"

    assert len(sync_calls) == 1
    assert sync_calls[0]["return_id"] == 1
    assert sync_calls[0]["decision"] == "Approved"