from types import SimpleNamespace

from planning_lab.algorithms.self_refine import deterministic_checks, reflect_and_refine


def good_draft() -> str:
    body = " ".join(
        [
            "Security awareness controls should include verification, "
            "reporting procedures, employee training, and follow-up checks."
        ]
        * 15
    )

    return f"# Security Awareness Plan\n- {body}"


class SelfRefineLLM:
    def __init__(self):
        self.calls = []

    def invoke(self, messages, **kwargs):
        self.calls.append(messages)

        system = messages[0][1]

        if "critique phase" in system:
            return SimpleNamespace(
                content="The draft is too short and lacks sufficient structure."
            )

        return SimpleNamespace(
            content=good_draft()
        )


def test_deterministic_checks_detect_short_unstructured_draft():
    issues = deterministic_checks(
        "Design a phishing awareness workshop",
        "Too short",
    )

    assert len(issues) >= 2


def test_self_refine_uses_critique_then_revision():
    llm = SelfRefineLLM()

    result = reflect_and_refine(
        "Create a structured security awareness plan",
        "Short draft",
        llm,
    )

    assert result.draft == "Short draft"

    assert result.critique == (
        "The draft is too short and lacks sufficient structure."
    )

    assert result.revised == good_draft()

    assert len(llm.calls) == 2


def test_self_refine_keeps_passing_draft_without_revision():
    class PassingLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages, **kwargs):
            self.calls += 1

            return SimpleNamespace(content="PASS")

    draft = good_draft()
    llm = PassingLLM()

    result = reflect_and_refine(
        "Create a structured security awareness plan",
        draft,
        llm,
    )

    assert result.critique == "PASS"
    assert result.revised == draft
    assert result.grounded_issues == []
    assert llm.calls == 1