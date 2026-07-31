import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db.init_db import initialize_database
from mcp_server.server import MarketLoopMCPServer
from mcp_server.tools.session import SessionContext, switch_active_user_role


class DummySession:
    def __init__(self) -> None:
        self.calls = 0

    async def send_tool_list_changed(self) -> None:
        self.calls += 1


class SessionToolTests(unittest.TestCase):
    def test_switch_active_user_role_updates_session_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "marketloop.db"
            initialize_database(db_path)
            connection = sqlite3.connect(db_path)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON;")

            with patch("mcp_server.tools.session.get_connection", return_value=connection):
                context = SessionContext()
                result = switch_active_user_role(3, context)

            self.assertEqual(result["status"], "role_changed")
            self.assertEqual(context.active_role, "Inventory Manager")
            connection.close()

    def test_server_switch_role_refreshes_visible_tools(self) -> None:
        server = MarketLoopMCPServer()
        server.register_modules()
        server.switch_role(3)
        self.assertIn("update_inventory_quantity", server.list_tools())

    def test_switch_role_notifies_session_when_tools_change(self) -> None:
        server = MarketLoopMCPServer()
        server.register_modules()
        server._session_context.active_role = "Customer Support"
        session = DummySession()
        server._session = session
        server.switch_role(3)
        self.assertEqual(session.calls, 1)


if __name__ == "__main__":
    unittest.main()
