from types import SimpleNamespace

from state_graph.core.models import GraphState, TransitionResult
from state_graph.core.engine import StateGraphEngine
from state_graph.core.transitions import TransitionTable
from state_graph.graph3.inventory_graph import InventoryGraph
from state_graph.hitl.policy import inventory_requires_human_intervention


class FakeHITL:
    def __init__(self):
        self.requests = {}

    def pause(self, state, *, reason):
        request_id = "inventory-request-1"
        self.requests[request_id] = {"decision": None, "state": state.model_dump(mode="json"), "reason": reason}
        return TransitionResult(next_node="inventory_react", status="waiting", updates={"waiting_request_id": request_id})

    def get_request(self, request_id):
        return self.requests.get(request_id)


class FakeQueue:
    def __init__(self):
        self.items = []

    def enqueue_hitl(self, **item):
        self.items.append(item)


def test_inventory_hitl_policy_uses_confidence_variance_and_policy():
    assert inventory_requires_human_intervention(confidence=0.69)
    assert inventory_requires_human_intervention(quantity_variance=10)
    assert inventory_requires_human_intervention(policy_violation=True)
    assert not inventory_requires_human_intervention(confidence=0.9, quantity_variance=2)


def test_inventory_graph_pauses_and_queues_admin_task(monkeypatch):
    queue = FakeQueue()
    graph = InventoryGraph(llm=object(), task_queue=queue)
    graph.hitl = FakeHITL()
    monkeypatch.setattr(
        "state_graph.graph3.inventory_graph.constrained_react",
        lambda **_kwargs: SimpleNamespace(
            success=False,
            output="Escalated to human review.",
            confidence=0.0,
            iterations=1,
            escalated=True,
        ),
    )
    state = GraphState(
        run_id="inventory-run-1",
        graph_name="inventory",
        current_node="inventory_react",
        goal="Warehouse count conflicts with the system count.",
        data={"quantity_variance": 12},
    )

    result = graph.inventory_react(state)

    assert result.status == "waiting"
    assert result.updates["waiting_request_id"] == "inventory-request-1"
    assert queue.items[0]["request_id"] == "inventory-request-1"
    assert queue.items[0]["state"]["data"]["quantity_variance"] == 12


def test_inventory_graph_resumes_with_admin_decision(monkeypatch):
    queue = FakeQueue()
    graph = InventoryGraph(llm=object(), task_queue=queue)
    graph.hitl = FakeHITL()
    calls = {"count": 0}

    def react(**_kwargs):
        calls["count"] += 1
        return SimpleNamespace(
            success=False,
            output="Escalated to human review.",
            confidence=0.0,
            iterations=1,
            escalated=True,
        )

    monkeypatch.setattr("state_graph.graph3.inventory_graph.constrained_react", react)

    class Store:
        def __init__(self):
            self.states = {}

        def save(self, state):
            self.states[state.run_id] = state.model_copy(deep=True)

        def load_latest(self, run_id):
            return self.states.get(run_id)

    store = Store()
    transitions = TransitionTable()
    transitions.add("inventory_react", "done")
    engine = StateGraphEngine(
        transitions=transitions,
        nodes=graph.nodes(),
        checkpoint_store=store,
        ticket_service=object(),
    )
    paused = engine.run(
        GraphState(
            run_id="inventory-run-2",
            graph_name="inventory",
            current_node="inventory_react",
            goal="Resolve warehouse discrepancy.",
        )
    )
    request_id = paused.waiting_request_id
    graph.hitl.requests[request_id]["decision"] = "approve"

    resumed = engine.resume("inventory-run-2")

    assert resumed.status == "done"
    assert resumed.outputs["hitl_decision"] == "approve"
    assert calls["count"] == 1
