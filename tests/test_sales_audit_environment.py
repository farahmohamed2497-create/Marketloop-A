import json

from planning_lab.sales_audit_environment import SalesAuditEnvironment


class FakeReportExecutor:
    def __init__(self):
        self.calls = []

    def execute(self, tool_name, arguments):
        self.calls.append((tool_name, arguments))
        return json.dumps(
            {
                "inventory": {
                    "low_stock_items": [
                        {"product_id": 4, "product_name": "Air Fryer", "quantity": 15}
                    ]
                }
            }
        )


def test_grounded_environment_accepts_authorized_low_stock_restock(monkeypatch):
    executor = FakeReportExecutor()
    environment = SalesAuditEnvironment(executor, "2026-01-01", "2026-01-31")

    class Connection:
        def execute(self, *_args):
            return self

        def fetchone(self):
            return {"role_name": "Inventory Manager"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr("planning_lab.sales_audit_environment.get_connection", Connection)

    feedback = environment.evaluate(
        json.dumps({"action": "restock", "product_id": 4, "quantity_change": 20, "user_id": 3})
    )

    assert feedback.success is True
    assert executor.calls == [
        ("generate_sales_audit_report", {"start_date": "2026-01-01", "end_date": "2026-01-31"})
    ]


def test_grounded_environment_catches_action_for_non_low_stock_product(monkeypatch):
    environment = SalesAuditEnvironment(FakeReportExecutor(), "2026-01-01", "2026-01-31")

    class Connection:
        def execute(self, *_args):
            return self

        def fetchone(self):
            return {"role_name": "Inventory Manager"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr("planning_lab.sales_audit_environment.get_connection", Connection)
    feedback = environment.evaluate(
        json.dumps({"action": "restock", "product_id": 1, "quantity_change": 20, "user_id": 3})
    )

    assert feedback.success is False
    assert "not low stock" in feedback.details[0]