from types import SimpleNamespace

from planning_lab.algorithms.decomposition import execute_plan
from planning_lab.models import EnvironmentFeedback, Plan


class GroundedLLM:
    def __init__(self):
        self.calls = []

    def invoke(self, messages, **kwargs):
        prompt = messages[-1][1]
        self.calls.append(prompt)

        if "External environment feedback:" in prompt:
            return SimpleNamespace(
                content=(
                    "# Improved Result\n"
                    "- Revised using the external evaluator feedback."
                )
            )

        return SimpleNamespace(
            content="Initial result that fails external validation."
        )


class SequencedEnvironment:
    def __init__(self, feedback):
        self.feedback = iter(feedback)
        self.states = []

    def evaluate(self, state):
        self.states.append(state)
        return next(self.feedback)


def make_plan():
    return Plan.model_validate(
        {
            "goal": "Create a grounded report",
            "tasks": [
                {
                    "id": "task1",
                    "instruction": "Create the report",
                    "depends_on": [],
                }
            ],
        }
    )


def test_grounding_retries_failed_subtask_using_feedback():
    llm = GroundedLLM()

    environment = SequencedEnvironment(
        [
            EnvironmentFeedback(
                success=False,
                score=0.2,
                details=["The result needs more concrete structure."],
            ),
            EnvironmentFeedback(
                success=True,
                score=0.95,
                details=[],
            ),
        ]
    )

    outputs = execute_plan(
        make_plan(),
        llm,
        environment=environment,
        max_grounding_retries=1,
    )

    assert outputs["task1"].startswith("# Improved Result")

    assert len(environment.states) == 2

    assert environment.states[0] == (
        "Initial result that fails external validation."
    )

    assert environment.states[1].startswith("# Improved Result")

    assert len(llm.calls) == 2

    assert "The result needs more concrete structure." in llm.calls[1]


def test_grounding_does_not_retry_successful_subtask():
    llm = GroundedLLM()

    environment = SequencedEnvironment(
        [
            EnvironmentFeedback(
                success=True,
                score=0.95,
                details=[],
            )
        ]
    )

    outputs = execute_plan(
        make_plan(),
        llm,
        environment=environment,
        max_grounding_retries=1,
    )

    assert outputs["task1"] == (
        "Initial result that fails external validation."
    )

    assert len(environment.states) == 1
    assert len(llm.calls) == 1