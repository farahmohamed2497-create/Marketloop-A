"""Tests that Self-Refine cannot ignore external validation feedback."""

from __future__ import annotations

from types import SimpleNamespace

from planning_lab.algorithms.self_refine import reflect_and_refine
from planning_lab.models import EnvironmentFeedback


class ScriptedLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.calls: list[object] = []

    def invoke(self, messages, **_kwargs):
        self.calls.append(messages)
        return SimpleNamespace(content=next(self.responses))


def test_grounded_failure_forces_revision_even_if_critic_says_pass() -> None:
    llm = ScriptedLLM(["PASS", "Corrected database-backed report."])

    result = reflect_and_refine(
        goal="Report January sales accurately.",
        draft="A deliberately incorrect report with enough words to look plausible. " * 4,
        llm=llm,
        grounded_check=lambda _draft: EnvironmentFeedback(
            success=False,
            score=0.0,
            details=["Revenue is incorrect; database value is 5000.00."],
        ),
        source_of_truth="MarketLoop SQLite database.",
    )

    assert result.critique == "PASS"
    assert result.revised == "Corrected database-backed report."
    assert any("Revenue is incorrect" in issue for issue in result.grounded_issues)
    assert len(llm.calls) == 2