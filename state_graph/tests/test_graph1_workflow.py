from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from planning_lab.algorithms.environment import Environment
from state_graph.core.models import GraphState
from state_graph.graph1.refund_graph import RefundGraph


def _state(node: str, **data):
    return GraphState(run_id="refund-flow", graph_name="refund", current_node=node,
                      goal="Refund a damaged order.", data=data)


def test_policy_grounding_then_warehouse_receipt_forms_a_real_cycle(monkeypatch):
    graph = RefundGraph(llm=MagicMock(), environment=Environment())
    monkeypatch.setattr("rag.rag_pipeline.answer_with_hybrid", lambda _: {"answer": "30-day return policy"})
    policy = graph.policy_grounding(_state("policy_grounding"))
    assert policy.next_node == "lats"
    assert policy.updates["data"]["policy_context"] == "30-day return policy"

    monkeypatch.setattr("state_graph.graph1.refund_graph.lats", lambda **_: SimpleNamespace(
        success=True, output="Refund after inspection", best_score=0.9, iterations=1
    ))
    decision = graph.lats_node(_state("lats", **policy.updates["data"]))
    wait = graph.evaluate_refund(_state("evaluate_refund", **decision.updates["data"]))
    assert wait.next_node == "awaiting_return"
    assert wait.status == "waiting"

    resume = graph.awaiting_return(_state("awaiting_return", **decision.updates["data"], return_received=True))
    assert resume.next_node == "evaluate_refund"
    done = graph.evaluate_refund(_state("evaluate_refund", **decision.updates["data"], return_received=True))
    assert done.status == "done"
