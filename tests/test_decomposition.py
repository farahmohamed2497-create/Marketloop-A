import pytest

from planning_lab.algorithms.decomposition import decompose_goal
from planning_lab.models import Plan


class FakeStructuredLLM:
    """Minimal LLM double for testing decomposition validation."""

    def __init__(self, generated_plan):
        self.generated_plan = generated_plan

    class StructuredRunner:
        def __init__(self, value):
            self.value = value

        def invoke(self, messages, **kwargs):
            return self.value

    def with_structured_output(self, schema, *, method):
        assert method == "json_schema"
        return self.StructuredRunner(self.generated_plan)


class FakeGeneratedPlan:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self):
        return self.payload


def test_valid_decomposition_builds_parallel_dag():
    llm = FakeStructuredLLM(
        FakeGeneratedPlan(
            {
                "goal": "ignored by caller",
                "tasks": [
                    {
                        "id": "t1",
                        "instruction": "Analyze sales performance",
                        "depends_on": [],
                    },
                    {
                        "id": "t2",
                        "instruction": "Analyze return performance",
                        "depends_on": [],
                    },
                    {
                        "id": "t3",
                        "instruction": "Analyze inventory risks",
                        "depends_on": [],
                    },
                    {
                        "id": "t4",
                        "instruction": "Synthesize management summary",
                        "depends_on": ["t1", "t2", "t3"],
                    },
                ],
            }
        )
    )

    goal = (
        "Analyze sales, returns, and inventory risks "
        "and produce a management summary"
    )

    plan = decompose_goal(goal, llm)

    assert plan.goal == goal

    assert plan.execution_batches() == [
        ["t1", "t2", "t3"],
        ["t4"],
    ]

    assert plan.topological_order()[-1] == "t4"

    assert plan.terminal_tasks() == ["t4"]


def test_missing_dependency_is_rejected():
    payload = {
        "goal": "Create a valid business analysis",
        "tasks": [
            {
                "id": "t1",
                "instruction": "Analyze sales",
                "depends_on": ["unknown"],
            }
        ],
    }

    with pytest.raises(ValueError, match="unknown dependencies"):
        Plan.model_validate(payload)


def test_duplicate_task_ids_are_rejected():
    payload = {
        "goal": "Create a valid business analysis",
        "tasks": [
            {
                "id": "t1",
                "instruction": "Analyze sales",
                "depends_on": [],
            },
            {
                "id": "t1",
                "instruction": "Analyze returns",
                "depends_on": [],
            },
        ],
    }

    with pytest.raises(ValueError, match="Task ids must be unique"):
        Plan.model_validate(payload)


def test_self_dependency_is_rejected():
    payload = {
        "goal": "Create a valid business analysis",
        "tasks": [
            {
                "id": "t1",
                "instruction": "Analyze sales",
                "depends_on": ["t1"],
            }
        ],
    }

    with pytest.raises(ValueError, match="cannot depend on itself"):
        Plan.model_validate(payload)


def test_cycle_is_rejected():
    payload = {
        "goal": "Create a valid business analysis",
        "tasks": [
            {
                "id": "t1",
                "instruction": "Analyze sales",
                "depends_on": ["t3"],
            },
            {
                "id": "t2",
                "instruction": "Analyze returns",
                "depends_on": ["t1"],
            },
            {
                "id": "t3",
                "instruction": "Analyze inventory",
                "depends_on": ["t2"],
            },
        ],
    }

    with pytest.raises(ValueError, match="Cycle detected"):
        Plan.model_validate(payload)