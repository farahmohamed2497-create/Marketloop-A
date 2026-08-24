"""e2e: HITL inbox and ticket board resolution through the admin API.

The state-graph engine itself (pausing, checkpointing, classifying
failures) is already covered by state_graph/tests/. What's new here is
the platform wiring: does hitting the HTTP endpoint actually record the
admin's decision AND push the run forward via engine_registry.resume_run,
the same way engine.resume() is proven to work in
tests/graph2_test/test_resume_after_ticket_resolution.py.

engine_registry.resume_run is faked here (no real LLM/graph execution) --
this test is about the endpoint -> HITLNode/FailureTicketService ->
resume_run call chain, not about re-testing the graphs themselves.
"""
from __future__ import annotations

import admin_api.main as admin_main


class _FakeGraphState:
    def __init__(self, run_id, status, current_node, outputs):
        self.run_id = run_id
        self.status = status
        self.current_node = current_node
        self.outputs = outputs


def test_hitl_task_appears_in_inbox_when_a_graph_pauses(client):
    hitl = admin_main.get_hitl()
    request_id = hitl.create_request(
        run_id="run-refund-1",
        graph_name="refund",
        reason="Refund exceeds the $500 auto-approval limit.",
        state={"current_node": "lats", "data": {"refund_amount": 900}},
    )

    response = client.get("/api/hitl/tasks", params={"status": "pending"})
    assert response.status_code == 200
    tasks = response.json()
    assert any(task["request_id"] == request_id for task in tasks)
    assert tasks[0]["reason"].startswith("Refund exceeds")


def test_admin_resolving_hitl_task_resumes_the_run_with_the_decision(client, monkeypatch):
    hitl = admin_main.get_hitl()
    request_id = hitl.create_request(
        run_id="run-refund-2",
        graph_name="refund",
        reason="Refund exceeds the $500 auto-approval limit.",
        state={"current_node": "lats"},
    )

    resume_calls = []

    def fake_resume_run(run_id, graph_name):
        resume_calls.append((run_id, graph_name))
        return _FakeGraphState(run_id, "done", "done", {"hitl_decision": "approve"})

    monkeypatch.setattr(admin_main, "resume_run", fake_resume_run)

    response = client.post(f"/api/hitl/tasks/{request_id}/resolve", json={"decision": "approve"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "done"
    assert body["outputs"]["hitl_decision"] == "approve"

    # The admin's decision must be durably recorded, not just passed
    # through in the HTTP response.
    stored = hitl.get_request(request_id)
    assert stored["decision"] == "approve"
    assert stored["status"] == "resolved"

    # And the *same* run must be the one that got resumed.
    assert resume_calls == [("run-refund-2", "refund")]


def test_resolving_unknown_hitl_task_is_a_404(client):
    response = client.post("/api/hitl/tasks/does-not-exist/resolve", json={"decision": "approve"})
    assert response.status_code == 404


def test_invalid_decision_is_rejected_without_resuming(client, monkeypatch):
    hitl = admin_main.get_hitl()
    request_id = hitl.create_request(
        run_id="run-refund-3", graph_name="refund", reason="test", state={}
    )

    resume_calls = []
    monkeypatch.setattr(
        admin_main, "resume_run", lambda run_id, graph_name: resume_calls.append((run_id, graph_name))
    )

    response = client.post(f"/api/hitl/tasks/{request_id}/resolve", json={"decision": "maybe"})

    assert response.status_code == 400
    assert resume_calls == []  # must not resume on a rejected/invalid decision


def test_ticket_appears_on_board_when_a_node_fails(client):
    tickets = admin_main.get_failure_tickets()
    ticket_id = tickets.create_ticket(
        run_id="run-shipping-1",
        graph_name="shipping",
        node_name="constrained_react",
        error="Carrier API returned a malformed tracking payload.",
        state={"current_node": "constrained_react"},
    )

    response = client.get("/api/tickets", params={"status": "open"})
    assert response.status_code == 200
    board = response.json()
    assert any(t["ticket_id"] == ticket_id for t in board)


def test_resolving_ticket_resumes_run_from_its_checkpoint(client, monkeypatch):
    tickets = admin_main.get_failure_tickets()
    ticket_id = tickets.create_ticket(
        run_id="run-shipping-2",
        graph_name="shipping",
        node_name="constrained_react",
        error="Carrier API returned a malformed tracking payload.",
        state={"current_node": "constrained_react"},
    )

    resume_calls = []

    def fake_resume_run(run_id, graph_name):
        resume_calls.append((run_id, graph_name))
        return _FakeGraphState(run_id, "done", "done", {"resolution": "Claim refiled."})

    monkeypatch.setattr(admin_main, "resume_run", fake_resume_run)

    response = client.patch(
        f"/api/tickets/{ticket_id}",
        json={"status": "resolved", "resolution": "Refiled the claim with corrected tracking data."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "resolved"
    assert body["resumed_run"]["status"] == "done"

    stored = tickets.get_ticket(ticket_id)
    assert stored["status"] == "resolved"
    assert resume_calls == [("run-shipping-2", "shipping")]


def test_moving_ticket_to_investigating_does_not_resume(client, monkeypatch):
    tickets = admin_main.get_failure_tickets()
    ticket_id = tickets.create_ticket(
        run_id="run-shipping-3", graph_name="shipping", node_name="constrained_react", error="boom"
    )

    resume_calls = []
    monkeypatch.setattr(
        admin_main, "resume_run", lambda run_id, graph_name: resume_calls.append((run_id, graph_name))
    )

    response = client.patch(f"/api/tickets/{ticket_id}", json={"status": "investigating"})

    assert response.status_code == 200
    assert "resumed_run" not in response.json()
    assert resume_calls == []  # investigating is not a resolution -- must not resume


def test_resolving_unknown_ticket_is_a_404(client):
    response = client.patch("/api/tickets/does-not-exist", json={"status": "resolved"})
    assert response.status_code == 404