from agent.planning_agent import SalesAuditPlanningAgent
from planning_lab.algorithms.decomposition import GeneratedPlan, PlannedTask


class FakeExecutor:
    def __init__(self):
        self.calls = []

    def execute(self, tool_name, arguments):
        self.calls.append((tool_name, arguments))
        return "grounded audit report"


def test_sales_audit_agent_routes_data_retrieval_to_existing_mcp_tool():
    executor = FakeExecutor()
    agent = SalesAuditPlanningAgent(
        object(),
        executor,
        start_date="2026-01-01",
        end_date="2026-01-31",
    )

    result = agent.solve("Retrieve audit data from the MCP report")

    assert result == "grounded audit report"
    assert executor.calls == [
        (
            "generate_sales_audit_report",
            {"start_date": "2026-01-01", "end_date": "2026-01-31"},
        )
    ]


class DecompositionLLM:
    class Runner:
        def invoke(self, _messages, **_kwargs):
            return GeneratedPlan(
                goal="ignored by adapter",
                tasks=[
                    PlannedTask(
                        id="audit",
                        instruction="Retrieve the January audit report.",
                        depends_on=[],
                        tool_name="generate_sales_audit_report",
                        tool_arguments={
                            "start_date": "2026-01-01",
                            "end_date": "2026-01-31",
                        },
                    )
                ],
            )

    def with_structured_output(self, _schema, **_kwargs):
        return self.Runner()

    def invoke(self, *_args, **_kwargs):
        raise AssertionError("The MCP-bound node should not call the LLM executor")


def test_sales_audit_agent_runs_toolkit_dag_through_mcp_adapter():
    executor = FakeExecutor()

    agent = SalesAuditPlanningAgent(
        DecompositionLLM(),
        executor,
        start_date="2026-01-01",
        end_date="2026-01-31",
    )

    run = agent.run_decomposition_first(
        "Retrieve the January MarketLoop audit report."
    )

    assert run.result == "grounded audit report"
    assert run.plan.topological_order() == ["audit"]
    assert executor.calls == [
        (
            "generate_sales_audit_report",
            {"start_date": "2026-01-01", "end_date": "2026-01-31"},
        )
    ]