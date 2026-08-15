"""Regression tests for the SQLite-backed Week 4 benchmark evaluator."""

from __future__ import annotations

import json

from planning_eval.grounded_environment import CaseGroundedEnvironment
from planning_eval.test_cases import get_case


def test_grounded_benchmark_accepts_database_truth() -> None:
    environment = CaseGroundedEnvironment(get_case("A1"))

    feedback = environment.evaluate(json.dumps(environment.truth))

    assert feedback.success is True
    assert feedback.score == 1.0


def test_grounded_benchmark_rejects_invented_audit() -> None:
    environment = CaseGroundedEnvironment(get_case("A1"))
    invented = (
        "Total revenue: 999999.00; Total orders: 999; Units sold: 999. "
        "Low-stock items: iPhone: 1 units."
    )

    feedback = environment.evaluate(invented)

    assert feedback.success is False
    assert any("Missing or incorrect" in detail for detail in feedback.details)
    assert any("Air Fryer" in detail for detail in feedback.details)