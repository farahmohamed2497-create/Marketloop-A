from types import SimpleNamespace

from planning_lab.algorithms.plan_and_solve import plan_and_solve


class FakePlanAndSolveLLM:
    class StructuredRunner:
        def __init__(self, owner):
            self.owner = owner

        def invoke(self, messages, **kwargs):
            self.owner.calls.append(messages)

            return SimpleNamespace(
                steps=[
                    "Inspect the supplied sales metrics.",
                    "Identify unusual operational patterns.",
                    "Summarize the highest-priority risks.",
                ]
            )

    def __init__(self):
        self.calls = []
        self.solve_prompt = None

    def with_structured_output(self, schema, *, method):
        assert method == "json_schema"
        return self.StructuredRunner(self)

    def invoke(self, messages, **kwargs):
        self.solve_prompt = messages[-1][1]

        return SimpleNamespace(
            content=(
                "The highest-priority risk is the observed return pattern. "
                "The recommendation is to investigate the affected orders "
                "and inventory before taking corrective action."
            )
        )


def test_plan_and_solve_has_explicit_plan_then_solve_phases():
    llm = FakePlanAndSolveLLM()

    result = plan_and_solve(
        "Analyze operational risks from the sales audit",
        llm,
        context='{"total_orders": 120, "total_returns": 18, "return_rate": 0.15}',
    )

    assert "highest-priority risk" in result

    assert len(llm.calls) == 1
    assert llm.solve_prompt is not None

    assert "Plan:" in llm.solve_prompt
    assert "1. Inspect the supplied sales metrics." in llm.solve_prompt
    assert "2. Identify unusual operational patterns." in llm.solve_prompt
    assert "3. Summarize the highest-priority risks." in llm.solve_prompt


def test_plan_and_solve_passes_mcp_context_to_both_phases():
    llm = FakePlanAndSolveLLM()

    context = (
        '{"total_revenue": 25000, '
        '"total_returns": 12, '
        '"low_stock_items": 4}'
    )

    plan_and_solve(
        "Assess business risks",
        llm,
        context=context,
    )

    planning_prompt = llm.calls[0][-1][1]

    assert context in planning_prompt
    assert context in llm.solve_prompt


def test_plan_and_solve_rejects_empty_plan():
    class EmptyPlanLLM(FakePlanAndSolveLLM):
        class StructuredRunner:
            def __init__(self, owner):
                self.owner = owner

            def invoke(self, messages, **kwargs):
                return SimpleNamespace(steps=[])

    llm = EmptyPlanLLM()

    try:
        plan_and_solve("Assess risks", llm)
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert "empty plan" in str(exc)