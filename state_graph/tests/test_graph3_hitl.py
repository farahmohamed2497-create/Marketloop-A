from types import SimpleNamespace

from state_graph.core.models import GraphState, TransitionResult
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
