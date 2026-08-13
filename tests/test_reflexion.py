from planning_lab.algorithms.reflexion import reflexion
from planning_lab.models import EnvironmentFeedback


class FakeEnvironment:
    def __init__(self, feedback):
        self.feedback = iter(feedback)

    def evaluate(self, _attempt):
        return next(self.feedback)


class FakeLLM:
    class Response:
        def __init__(self, content):
            self.content = content

    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts: list[str] = []

    def invoke(self, messages, **_kwargs):
        self.prompts.append(messages[-1][1])
        return self.Response(next(self.responses))


def test_reflexion_carries_grounded_lesson_to_the_next_sales_audit_trial():
    llm = FakeLLM(
        [
            "Restock product 4 without checking the quantity.",
            "I must check the low-stock evidence and manager approval before restocking.",
            "Restock product 4 only after confirming its low-stock evidence and approval.",
        ]
    )
    environment = FakeEnvironment(
        [
            EnvironmentFeedback(
                success=False,
                score=0.2,
                details=["The proposed restock is missing evidence and manager approval."],
            ),
            EnvironmentFeedback(success=True, score=1.0),
        ]
    )

    result = reflexion(
        "Choose a safe sales-audit inventory action.",
        llm,
        environment,
        max_trials=2,
        memory_size=2,
    )

    assert result.success is True
    assert result.trials[0].reflection.startswith("I must check")
    assert "I must check the low-stock evidence" in llm.prompts[2]


def test_reflexion_buffer_discards_old_lessons_when_full():
    llm = FakeLLM(
        [
            "attempt one", "I learned one",
            "attempt two", "I learned two",
            "attempt three", "I learned three",
        ]
    )
    environment = FakeEnvironment(
        [
            EnvironmentFeedback(success=False, score=0.1),
            EnvironmentFeedback(success=False, score=0.2),
            EnvironmentFeedback(success=False, score=0.3),
        ]
    )

    result = reflexion("Validate a sales-audit action.", llm, environment, max_trials=3, memory_size=2)

    assert result.success is False
    assert result.memory == ["I learned two", "I learned three"]