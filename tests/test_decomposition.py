import pytest

from planning_lab.algorithms.decomposition import decompose_goal, execute_plan
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


def test_sales_audit_decomposition_builds_parallel_dag():
    llm = FakeStructuredLLM(
        FakeGeneratedPlan(
            {
                "goal": "ignored by caller",
                "tasks": [
                    {
                        "id": "sales",
                        "instruction": "Analyze January 2026 sales performance and revenue trends.",
                        "depends_on": [],
                    },
                    {
                        "id": "returns",
                        "instruction": "Calculate the January 2026 return rate and return drivers.",
                        "depends_on": [],
                    },
                    {
                        "id": "inventory",
                        "instruction": "Identify low-stock products and their operational exposure.",
                        "depends_on": [],
                    },
                    {
                        "id": "audit_log",
                        "instruction": "Review operational audit-log activity for the reporting period.",
                        "depends_on": [],
                    },
                    {
                        "id": "risk",
                        "instruction": "Identify actionable operational risks from returns and low stock.",
                        "depends_on": ["returns", "inventory", "audit_log"],
                    },
                    {
                        "id": "action",
                        "instruction": "Recommend a concrete operational action to reduce the highest risk.",
                        "depends_on": ["sales", "risk"],
                    },
                    {
                        "id": "summary",
                        "instruction": "Produce a management summary with the recommended operational action.",
                        "depends_on": ["action"],
                    },
                ],
            }
        )
    )

    goal = (
        "Analyze January 2026 sales performance, check return rate and low-stock "
        "products, identify operational risks, recommend an action, and produce a final management summary."
    )

    plan = decompose_goal(goal, llm)

    assert plan.goal == goal

    assert plan.execution_batches() == [
        ["audit_log", "inventory", "returns", "sales"],
        ["risk"],
        ["action"],
        ["summary"],
    ]

    assert plan.topological_order()[-1] == "summary"

    assert plan.terminal_tasks() == ["summary"]


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


class FakeExecutionLLM:
    class Response:
        def __init__(self, content):
            self.content = content

    def __init__(self):
        self.prompts: list[str] = []

    def invoke(self, messages, **_kwargs):
        prompt = messages[-1][1]
        self.prompts.append(prompt)
        return self.Response(f"completed node {len(self.prompts)}")


def test_executor_passes_dependency_outputs_to_sales_audit_synthesis():
    plan = Plan.model_validate(
        {
            "goal": "Create an actionable sales audit for January 2026.",
            "tasks": [
                {
                    "id": "sales",
                    "instruction": "Analyze sales performance for January 2026.",
                    "depends_on": [],
                },
                {
                    "id": "summary",
                    "instruction": "Write the final management summary.",
                    "depends_on": ["sales"],
                },
            ],
        }
    )
    llm = FakeExecutionLLM()

    outputs = execute_plan(plan, llm, max_workers=1)

    assert outputs == {"sales": "completed node 1", "summary": "completed node 2"}
    assert "OUTPUT FROM sales:\ncompleted node 1" in llm.prompts[1]