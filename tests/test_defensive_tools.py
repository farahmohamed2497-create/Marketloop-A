import asyncio
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db.init_db import initialize_database
from mcp_server.tools.inventory import update_inventory_quantity
from mcp_server.tools.orders import process_return_request


class DummySession:
    def __init__(self, action: str) -> None:
        self.action = action
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def elicit_form(self, message: str, requested_schema: dict[str, object], related_request_id: int | None = None) -> object:
        self.calls.append((message, requested_schema))
        return SimpleNamespace(action=self.action, content=None)


class DefensiveToolTests(unittest.TestCase):
    def test_process_return_request_for_delivered_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "marketloop.db"
            initialize_database(db_path)
            connection = sqlite3.connect(db_path)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON;")

            with patch("mcp_server.tools.orders.get_connection", return_value=connection):
                result = asyncio.run(
                    process_return_request(
                        {
                            "order_id": 1,
                            "customer_id": 1,
                            "reason": "The item did not match the description",
                        },
                        context=SimpleNamespace(session=DummySession("accept")),
                    )
                )

            self.assertEqual(result["status"], "Pending")
            connection.close()

    def test_process_return_request_can_be_cancelled_by_admin(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "marketloop.db"
            initialize_database(db_path)
            connection = sqlite3.connect(db_path)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON;")

            with patch("mcp_server.tools.orders.get_connection", return_value=connection):
                result = asyncio.run(
                    process_return_request(
                        {
                            "order_id": 1,
                            "customer_id": 1,
                            "reason": "The item did not match the description",
                        },
                        context=SimpleNamespace(session=DummySession("decline")),
                    )
                )

            self.assertEqual(result["status"], "cancelled")
            self.assertIn("cancelled", result["message"].lower())
            connection.close()

    def test_update_inventory_quantity_requires_authorized_user(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "marketloop.db"
            initialize_database(db_path)
            connection = sqlite3.connect(db_path)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON;")

            with patch("mcp_server.tools.inventory.get_connection", return_value=connection):
                result = update_inventory_quantity(
                    {
                        "product_id": 1,
                        "quantity_change": 5,
                        "user_id": 3,
                    }
                )

            self.assertEqual(result["new_quantity"], 25)
            connection.close()


if __name__ == "__main__":
    unittest.main()
