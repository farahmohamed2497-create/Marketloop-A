from __future__ import annotations

from state_graph.core.models import GraphState
from state_graph.hitl.node import HITLNode


def _make_request():
    node = HITLNode()

    state = GraphState(
        run_id="platform-hitl-run",
        graph_name="dispute",
        current_node="dispute_react",
        goal="Review this dispute.",
    )

    paused = node.pause(
        state,
        reason="Admin review required.",
    )

    return node, paused


def test_hitl_task_appears_on_platform_when_graph_pauses(
    platform_env,
):
    from fastapi.testclient import TestClient
    import admin_api.main as api

    api._hitl_service = None

    node, paused = _make_request()

    client = TestClient(api.app)

    response = client.get(
        "/api/hitl/tasks"
    )

    assert response.status_code == 200

    tasks = response.json()["tasks"]

    assert any(
        task["request_id"] == paused.updates["waiting_request_id"]
        for task in tasks
    )


def test_admin_resolves_hitl_and_resume_is_triggered(
    platform_env,
    monkeypatch,
):
    from fastapi.testclient import TestClient
    import admin_api.main as api

    api._hitl_service = None

    node, paused = _make_request()
    request_id = paused.updates["waiting_request_id"]

    calls = []

    def fake_resume_run(*, run_id, graph_name):
        calls.append(
            {
                "run_id": run_id,
                "graph_name": graph_name,
            }
        )
        return {
            "status": "done",
            "run_id": run_id,
        }

    monkeypatch.setattr(
        api,
        "resume_run",
        fake_resume_run,
    )

    client = TestClient(api.app)

    response = client.post(
        f"/api/hitl/tasks/{request_id}/resolve",
        json={"decision": "approve"},
    )

    assert response.status_code == 200

    assert calls == [
        {
            "run_id": "platform-hitl-run",
            "graph_name": "dispute",
        }
    ]

    saved = node.get_request(request_id)

    assert saved["decision"] == "approve"
    assert saved["status"] == "resolved"