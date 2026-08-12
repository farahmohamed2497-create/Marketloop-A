from planning_lab.evaluation import (
    calculate_metrics,
    evaluate_decomposition,
    evaluate_plan_and_solve,
    evaluate_self_refine,
)
from planning_lab.models import Plan


def test_calculate_metrics():
    metrics = calculate_metrics(
        [True, True, False, True]
    )

    assert metrics.total == 4
    assert metrics.passed == 3
    assert metrics.failed == 1
    assert metrics.pass_rate == 0.75
    assert metrics.success is False


def test_empty_metrics_are_safe():
    metrics = calculate_metrics([])

    assert metrics.total == 0
    assert metrics.passed == 0
    assert metrics.failed == 0
    assert metrics.pass_rate == 0.0


def test_decomposition_evaluation():
    plan = Plan.model_validate(
        {
            "goal": "Prepare a launch brief",
            "tasks": [
                {
                    "id": "research",
                    "instruction": "Research the audience",
                    "depends_on": [],
                },
                {
                    "id": "risks",
                    "instruction": "Identify launch risks",
                    "depends_on": [],
                },
                {
                    "id": "brief",
                    "instruction": "Synthesize the launch brief",
                    "depends_on": ["research", "risks"],
                },
            ],
        }
    )

    result = evaluate_decomposition(plan)

    assert result["component"] == "decomposition"
    assert result["execution_batches"] == [
        ["research", "risks"],
        ["brief"],
    ]
    assert result["terminal_tasks"] == ["brief"]

    metrics = result["metrics"]

    assert metrics.total == 3
    assert metrics.passed == 3
    assert metrics.pass_rate == 1.0


def test_plan_and_solve_evaluation():
    output = """
    PLAN
    1. Understand the requirements.
    2. Analyze the available information.

    SOLUTION
    The final answer follows from the plan.
    """

    result = evaluate_plan_and_solve(output)

    assert result["component"] == "plan_and_solve"

    metrics = result["metrics"]

    assert metrics.total == 3
    assert metrics.passed == 3
    assert metrics.pass_rate == 1.0


def test_self_refine_evaluation_requires_revision_when_issues_exist():
    result = evaluate_self_refine(
        draft="Short draft",
        revised="# Improved\n- Detailed revised deliverable",
        grounded_issues=["Draft is too short."],
    )

    assert result["component"] == "self_refine"
    assert result["draft_changed"] is True
    assert result["grounded_issue_count"] == 1

    metrics = result["metrics"]

    assert metrics.total == 3
    assert metrics.passed == 3
    assert metrics.pass_rate == 1.0


def test_self_refine_evaluation_keeps_clean_draft():
    draft = "# Complete\n- A sufficiently structured deliverable"

    result = evaluate_self_refine(
        draft=draft,
        revised=draft,
        grounded_issues=[],
    )

    assert result["draft_changed"] is False
    assert result["grounded_issue_count"] == 0

    metrics = result["metrics"]

    assert metrics.success is True