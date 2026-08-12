from planning_lab.evaluation import (
    evaluate_decomposition,
    evaluate_plan_and_solve,
    evaluate_self_refine,
)
from planning_lab.models import Plan


def test_decomposition_evaluation_integrates_with_plan():
    plan = Plan.model_validate(
        {
            "goal": "Create an integrated report",
            "tasks": [
                {
                    "id": "research",
                    "instruction": "Research the topic",
                    "depends_on": [],
                },
                {
                    "id": "synthesis",
                    "instruction": "Synthesize the report",
                    "depends_on": ["research"],
                },
            ],
        }
    )

    evaluation = evaluate_decomposition(plan)

    assert evaluation["component"] == "decomposition"
    assert evaluation["terminal_tasks"] == ["synthesis"]
    assert evaluation["metrics"].success is True


def test_plan_and_solve_evaluation_integrates_with_output():
    output = """
    PLAN
    1. Understand the problem.
    2. Solve it step by step.

    SOLUTION
    The problem is solved using the planned steps.
    """

    evaluation = evaluate_plan_and_solve(output)

    assert evaluation["component"] == "plan_and_solve"
    assert evaluation["metrics"].success is True


def test_self_refine_evaluation_integrates_with_revision():
    draft = "Initial draft"
    revised = "# Improved Result\n- Better structured deliverable"

    evaluation = evaluate_self_refine(
        draft=draft,
        revised=revised,
        grounded_issues=["The draft needs more structure."],
    )

    assert evaluation["component"] == "self_refine"
    assert evaluation["draft_changed"] is True
    assert evaluation["grounded_issue_count"] == 1
    assert evaluation["metrics"].success is True