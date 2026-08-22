"""Unit tests for Graph 3 (Retention) state transitions.

Ticket item: `test(graph3): unit tests for state transitions`

`constrained_react` is monkeypatched everywhere so these tests exercise
OUR transition logic (which target node, which branch, what gets
persisted) deterministically, without depending on real model output.
"""
from __future__ import annotations

import sys
import pathlib
from unittest.mock import MagicMock

import pytest

# --- make both the deliverable package and the local repo stubs importable ---
# NOTE: this sys.path setup is sandbox-only scaffolding so these tests can
# run standalone in isolation. In the real repo, graph3's package already
# sits under the project root next to `state_graph/`, `mcp_server/`, and
# `planning_lab/`, so none of this is needed there -- just
# `from graph3.state import RetentionNode` etc.
_PKG_ROOT = pathlib.Path(__file__).resolve().parent.parent          # .../retention
_PROJECT_ROOT = _PKG_ROOT.parent                                     # .../  (parent of the package)
sys.path.insert(0, str(_PROJECT_ROOT))              # so `retention.state` etc. import as a package
sys.path.insert(0, str(_PKG_ROOT / "_repo_stubs"))  # so `state_graph`, `mcp_server`, `planning_lab` import

import planning_lab.algorithms.react as react_mod  # noqa: E402
from state_graph.core.engine import StateGraphEngine  # noqa: E402
from state_graph.core.models import GraphState  # noqa: E402
from state_graph.core.transitions import TransitionTable  # noqa: E402
from state_graph.hitl.node import HITLNode  # noqa: E402

from graph3.state import RetentionNode, build_node_registry  # noqa: E402
from graph3.retention_graph import RetentionGraph, MAX_NEGOTIATION_ROUNDS  # noqa: E402
from graph3.graph import build_graph3, create_initial_state  # noqa: E402


@pytest.fixture
def graph():
    return RetentionGraph(llm=MagicMock())


@pytest.fixture
def engine(graph):
    transitions = TransitionTable()
    transitions.add(RetentionNode.AWAITING_INPUT.value, RetentionNode.POLICY_LOOKUP.value)
    transitions.add(RetentionNode.POLICY_LOOKUP.value, RetentionNode.RETENTION_REACT.value)
    transitions.add(RetentionNode.RETENTION_REACT.value, RetentionNode.AWAITING_CUSTOMER_RESPONSE.value)
    transitions.add(RetentionNode.RETENTION_REACT.value, RetentionNode.DONE.value)
    transitions.add(RetentionNode.AWAITING_CUSTOMER_RESPONSE.value, RetentionNode.RETENTION_REACT.value)
    return StateGraphEngine(transitions=transitions, nodes=graph.nodes())


def _react_result(**overrides):
    result = MagicMock()
    result.success = True
    result.output = "stub"
    result.confidence = 0.95
    result.iterations = 1
    result.escalated = False
    for k, v in overrides.items():
        setattr(result, k, v)
    return result


def base_state(**data_overrides) -> GraphState:
    data = {
        "subscription_id": 1,
        "retention_offer_value": 50.0,
        "legal_threat": False,
        "policy_violation": False,
        "proposed_action": "send_retention_offer",
        "negotiation_round": 0,
    }
    data.update(data_overrides)
    return GraphState(
        run_id="run-1", graph_name="retention",
        current_node=RetentionNode.AWAITING_INPUT.value,
        goal="customer wants to cancel", data=data,
    )


# --------------------------------------------------------------------- #
# awaiting_input
# --------------------------------------------------------------------- #

def test_awaiting_input_blocks_on_empty_goal(graph):
    state = base_state()
    state = state.model_copy(update={"goal": ""})
    result = graph.awaiting_input(state)
    assert result.next_node == RetentionNode.AWAITING_INPUT.value
    assert result.status == "waiting"


def test_awaiting_input_advances_to_policy_lookup(graph):
    result = graph.awaiting_input(base_state())
    assert result.next_node == RetentionNode.POLICY_LOOKUP.value
    assert result.status is None


# --------------------------------------------------------------------- #
# policy_lookup (RAG seam — stub today)
# --------------------------------------------------------------------- #

def test_policy_lookup_advances_to_retention_react(graph):
    result = graph.policy_lookup(base_state())
    assert result.next_node == RetentionNode.RETENTION_REACT.value
    assert "policy_context" in result.updates["data"]


# --------------------------------------------------------------------- #
# retention_react (Constrained ReAct — conditional branching)
# --------------------------------------------------------------------- #

def test_retention_react_offer_goes_to_awaiting_customer(monkeypatch, graph):
    monkeypatch.setattr(
        "graph3.retention_graph.constrained_react",
        lambda **kw: _react_result(confidence=0.95),
    )
    state = base_state(proposed_action="send_retention_offer", retention_offer_value=50.0)
    result = graph.retention_react(state)
    assert result.next_node == RetentionNode.AWAITING_CUSTOMER_RESPONSE.value
    assert result.status == "waiting"
    assert result.updates["data"]["negotiation_round"] == 1


def test_retention_react_direct_discount_within_policy_goes_done(monkeypatch, graph):
    monkeypatch.setattr(
        "graph3.retention_graph.constrained_react",
        lambda **kw: _react_result(confidence=0.95),
    )
    state = base_state(
        proposed_action="apply_discount_code",
        retention_offer_value=50.0,       # dollar figure, HITL threshold check
        proposed_discount_pct=0.15,       # separate fraction, what actually gets applied
    )
    result = graph.retention_react(state)
    assert result.next_node == RetentionNode.DONE.value
    assert result.status == "done"
    # Not just the status -- verify the tool actually ran and wrote the DB,
    # not merely that the graph claims it did.
    assert result.updates["outputs"]["resolution"]["synced"] is True
    from mcp_server.db import get_connection
    with get_connection() as conn:
        row = conn.execute(
            "SELECT status, discount_pct FROM Subscriptions WHERE subscription_id=1"
        ).fetchone()
    assert row["status"] == "retained"
    assert row["discount_pct"] == pytest.approx(0.15)


def test_retention_react_legal_threat_pauses_for_hitl(monkeypatch, graph):
    monkeypatch.setattr(
        "graph3.retention_graph.constrained_react",
        lambda **kw: _react_result(confidence=0.95),
    )
    state = base_state(legal_threat=True)
    result = graph.retention_react(state)
    # Self-loop: same node, status waiting, a durable HITL request created.
    assert result.next_node == RetentionNode.RETENTION_REACT.value
    assert result.status == "waiting"
    assert result.updates["waiting_request_id"]
    persisted = graph.hitl.get_request(result.updates["waiting_request_id"])
    assert persisted["status"] == "pending"
    assert persisted["reason"].startswith("Retention offer requires")


def test_retention_react_high_value_offer_requires_hitl(monkeypatch, graph):
    monkeypatch.setattr(
        "graph3.retention_graph.constrained_react",
        lambda **kw: _react_result(confidence=0.95),
    )
    state = base_state(retention_offer_value=999.0)  # above $500 cap
    result = graph.retention_react(state)
    assert result.next_node == RetentionNode.RETENTION_REACT.value
    assert result.status == "waiting"


def test_retention_react_low_confidence_requires_hitl(monkeypatch, graph):
    monkeypatch.setattr(
        "graph3.retention_graph.constrained_react",
        lambda **kw: _react_result(confidence=0.3),
    )
    result = graph.retention_react(base_state())
    assert result.status == "waiting"


def test_retention_react_resumes_after_hitl_pending_stays_waiting(monkeypatch, graph):
    monkeypatch.setattr(
        "graph3.retention_graph.constrained_react",
        lambda **kw: _react_result(confidence=0.95),
    )
    paused = graph.retention_react(base_state(legal_threat=True))
    request_id = paused.updates["waiting_request_id"]

    state = base_state(hitl_request_id=request_id)
    result = graph.retention_react(state)
    # Admin hasn't decided yet -> still waiting on the same request.
    assert result.status == "waiting"
    assert result.updates["waiting_request_id"] == request_id


def test_retention_react_resumes_after_hitl_approval_applies_and_finishes(monkeypatch, graph):
    monkeypatch.setattr(
        "graph3.retention_graph.constrained_react",
        lambda **kw: _react_result(confidence=0.95),
    )
    paused = graph.retention_react(base_state(legal_threat=True))
    request_id = paused.updates["waiting_request_id"]
    graph.hitl.resolve(request_id, "approve")

    state = base_state(hitl_request_id=request_id, proposed_discount_pct=0.2)
    result = graph.retention_react(state)
    assert result.next_node == RetentionNode.DONE.value
    assert result.status == "done"
    assert result.updates["data"]["hitl_decision"] == "approve"
    assert result.updates["outputs"]["resolution"]["discount_pct"] == pytest.approx(0.2)


def test_retention_react_resumes_after_hitl_rejection_churns(monkeypatch, graph):
    monkeypatch.setattr(
        "graph3.retention_graph.constrained_react",
        lambda **kw: _react_result(confidence=0.95),
    )
    paused = graph.retention_react(base_state(legal_threat=True))
    request_id = paused.updates["waiting_request_id"]
    graph.hitl.resolve(request_id, "reject")

    state = base_state(hitl_request_id=request_id)
    result = graph.retention_react(state)
    assert result.next_node == RetentionNode.DONE.value
    assert result.updates["outputs"]["resolution"] == "churned"


# --------------------------------------------------------------------- #
# awaiting_customer_response — the genuine cycle
# --------------------------------------------------------------------- #

def test_awaiting_customer_response_no_reply_stays_waiting(graph):
    state = base_state(negotiation_round=1, customer_response=None)
    result = graph.awaiting_customer_response(state)
    assert result.next_node == RetentionNode.AWAITING_CUSTOMER_RESPONSE.value
    assert result.status == "waiting"


def test_awaiting_customer_response_accept_finishes(graph):
    state = base_state(negotiation_round=1, customer_response="accept")
    result = graph.awaiting_customer_response(state)
    assert result.next_node == RetentionNode.DONE.value
    assert result.updates["outputs"]["resolution"] == "retained"


def test_awaiting_customer_response_reject_cycles_back(graph):
    state = base_state(negotiation_round=1, customer_response="reject")
    result = graph.awaiting_customer_response(state)
    assert result.next_node == RetentionNode.RETENTION_REACT.value
    assert result.updates["data"]["customer_response"] is None


def test_awaiting_customer_response_rounds_exhausted_flags_policy_violation(graph):
    state = base_state(negotiation_round=MAX_NEGOTIATION_ROUNDS, customer_response="reject")
    result = graph.awaiting_customer_response(state)
    assert result.next_node == RetentionNode.RETENTION_REACT.value
    assert result.updates["data"]["policy_violation"] is True


# --------------------------------------------------------------------- #
# End-to-end: a real multi-round cycle through the engine
# --------------------------------------------------------------------- #

def test_full_negotiation_cycle_visits_retention_react_more_than_once(monkeypatch, engine):
    monkeypatch.setattr(
        "graph3.retention_graph.constrained_react",
        lambda **kw: _react_result(confidence=0.95),
    )
    state = create_initial_state(run_id="run-42", goal="cancel please", subscription_id=7)
    state = engine.run(state)  # runs to first "waiting" (offer sent, awaiting customer)
    assert state.current_node == RetentionNode.AWAITING_CUSTOMER_RESPONSE.value

    visited = [state.current_node]
    for _ in range(2):
        state = state.model_copy(update={
            "status": "running",
            "data": {**state.data, "customer_response": "reject"},
        })
        state = engine.run(state)
        visited.append(state.current_node)

    # retention_react was revisited more than once with new information
    # each time (the rejections) -- a DAG cannot express this.
    assert visited.count(RetentionNode.AWAITING_CUSTOMER_RESPONSE.value) >= 2


def test_engine_rejects_invalid_transition(engine):
    from state_graph.core.exceptions import InvalidTransitionError
    with pytest.raises(InvalidTransitionError):
        engine.transitions.validate(
            RetentionNode.AWAITING_INPUT.value, RetentionNode.DONE.value
        )