"""Tests for the real-metrics comparison-table generator."""

from __future__ import annotations

from planning_eval.summarize_results import (
    EXPECTED_METHODS,
    comparison_rows,
    coverage_errors,
    render_markdown,
)
from planning_eval.test_cases import TEST_CASES


def _record(case_id: str, category: str, method: str) -> dict:
    return {
        "case_id": case_id,
        "category": category,
        "method": method,
        "success": True,
        "grounded": method != "lats_ungrounded",
        "llm_calls": 2,
        "total_tokens": 100,
        "latency_s": 0.5,
        "cost_usd": 0.001,
    }


def test_summary_rejects_missing_required_method() -> None:
    results = [_record("C1", "LOOKAHEAD", "plan_and_solve")]

    errors = coverage_errors(results)

    assert any("lats" in error for error in errors)


def test_summary_renders_measured_rows() -> None:
    results = [
        _record(case.id, case.category, method)
        for case in TEST_CASES
        for method in EXPECTED_METHODS[case.category]
    ]

    assert coverage_errors(results) == []
    table = render_markdown(comparison_rows(results))
    assert "| lats | True |" in table
    assert "| lats_ungrounded | False |" in table