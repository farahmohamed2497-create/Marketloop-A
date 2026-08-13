import pytest

from planning_lab.algorithms.tree_of_thoughts import (
    ThoughtCandidates,
    ThoughtEvaluation,
    tree_of_thoughts,
)


class FakeStructuredLLM:
    def __init__(self, candidate_batches, evaluations):
        self.candidate_batches = iter(candidate_batches)
        self.evaluations = iter(evaluations)
        self.generated_paths: list[str] = []

    class Runner:
        def __init__(self, callback):
            self.callback = callback

        def invoke(self, messages, **_kwargs):
            return self.callback(messages)

    def with_structured_output(self, schema, *, method):
        assert method == "json_schema"
        if schema is ThoughtCandidates:
            def generate(messages):
                self.generated_paths.append(messages[-1][1])
                return ThoughtCandidates(candidates=next(self.candidate_batches))

            return self.Runner(generate)

        if schema is ThoughtEvaluation:
            return self.Runner(
                lambda _messages: ThoughtEvaluation(
                    score=next(self.evaluations), rationale="test score"
                )
            )
        raise AssertionError(f"Unexpected schema: {schema}")


def test_bfs_prunes_low_scoring_sales_audit_branches():
    llm = FakeStructuredLLM(
        candidate_batches=[["Restock low-stock Air Fryer", "Ignore stock risk"]],
        evaluations=[0.92, 0.20],
    )

    thoughts = tree_of_thoughts(
        "Choose an operational action after a MarketLoop sales audit.",
        llm,
        depth=1,
        search_strategy="bfs",
        prune_threshold=0.50,
    )

    assert [thought.state for thought in thoughts] == ["Restock low-stock Air Fryer"]


def test_dfs_follows_highest_scoring_branch_before_siblings():
    llm = FakeStructuredLLM(
        candidate_batches=[
            ["Restock Air Fryer", "Offer discount"],
            ["Approve 20-unit restock", "Approve 10-unit restock"],
            ["Launch 10% discount", "Launch 5% discount"],
        ],
        evaluations=[0.90, 0.70, 0.95, 0.80, 0.85, 0.75],
    )

    thoughts = tree_of_thoughts(
        "Select an approved sales-audit action.",
        llm,
        depth=2,
        beam_width=2,
        search_strategy="dfs",
    )

    assert thoughts[0].state == "Approve 20-unit restock"
    assert "Partial path: Restock Air Fryer" in llm.generated_paths[1]


def test_tot_rejects_unknown_search_strategy():
    with pytest.raises(ValueError, match="search_strategy"):
        tree_of_thoughts("Audit January sales.", object(), search_strategy="random")