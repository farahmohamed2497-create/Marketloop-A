from planning_lab.algorithms.decomposition import execute_plan
from planning_lab.mcp_executor import MarketLoopMCPExecutor
from planning_lab.models import Plan
import pytest


class FailingLLM:
    def invoke(self, *_args, **_kwargs):
        raise AssertionError("Tool-bound nodes must use the MCP executor.")


class RecordingMCPExecutor:
    def __init__(self):
        self.calls: list[tuple[str, dict[str, object]]] = []

    def execute(self, tool_name: str, arguments: dict[str, object]) -> str:
        self.calls.append((tool_name, arguments))
        return f"{tool_name} completed"


def test_decomposition_first_executes_sales_audit_tool_nodes():
    plan = Plan.model_validate(
        {
            "goal": "Audit sales and approve a restock action for January 2026.",
            "tasks": [
                {
                    "id": "audit_data",
                    "instruction": "Retrieve the January sales audit data from MarketLoop.",
                    "depends_on": [],
                    "tool_name": "generate_sales_audit_report",
                    "tool_arguments": {
                        "start_date": "2026-01-01",
                        "end_date": "2026-01-31",
                    },
                },
                {
                    "id": "restock",
                    "instruction": "Apply the approved restock adjustment for the selected product.",
                    "depends_on": ["audit_data"],
                    "tool_name": "update_inventory_quantity",
                    "tool_arguments": {
                        "product_id": 4,
                        "quantity_change": 20,
                        "user_id": 3,
                    },
                },
            ],
        }
    )
    executor = RecordingMCPExecutor()

    outputs = execute_plan(plan, FailingLLM(), task_executor=executor)

    assert executor.calls == [
        (
            "generate_sales_audit_report",
            {"start_date": "2026-01-01", "end_date": "2026-01-31"},
        ),
        (
            "update_inventory_quantity",
            {"product_id": 4, "quantity_change": 20, "user_id": 3},
        ),
    ]
    assert outputs["restock"] == "update_inventory_quantity completed"


def test_mutating_mcp_tool_requires_explicit_manager_approval():
    executor = MarketLoopMCPExecutor(
        tools={"update_inventory_quantity": lambda _payload: {"status": "updated"}}
    )

    with pytest.raises(PermissionError, match="allow_mutations=True"):
        executor.execute(
            "update_inventory_quantity",
            {"product_id": 4, "quantity_change": 20, "user_id": 3},
        )