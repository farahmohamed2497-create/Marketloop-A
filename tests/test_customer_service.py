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
from mcp_server.server import MarketLoopMCPServer
from mcp_server.tools.customer_service import (
    SamplingUnavailableError,
    ToolValidationError,
    generate_delay_apology,
)


class DummySession:
    def __init__(self, text: str = "Generated apology email text") -> None:
        self.text = text
        self.calls: list[dict[str, object]] = []

    async def create_message(
        self,
        messages,
        *,
        max_tokens: int,
        system_prompt: str | None = None,
        include_context=None,
        temperature: float | None = None,
        stop_sequences=None,
        metadata=None,
        model_preferences=None,
        tools=None,
        tool_choice=None,
        related_request_id=None,
    ) -> object:
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "related_request_id": related_request_id,
            }
        )
        return SimpleNamespace(content=SimpleNamespace(text=self.text))


class CustomerServiceToolTests(unittest.TestCase):
    def test_generate_delay_apology_uses_sampling_with_order_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "marketloop.db"
            initialize_database(db_path)
            connection = sqlite3.connect(db_path)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON;")
            session = DummySession()
            try:
                with patch("mcp_server.tools.customer_service.get_connection", return_value=connection):
                    result = asyncio.run(
                        generate_delay_apology(
                            {"order_id": 1},
                            context=SimpleNamespace(session=session, request_id="req-1"),
                        )
                    )
            finally:
                connection.close()

            self.assertEqual(result, "Generated apology email text")
            self.assertEqual(len(session.calls), 1)
            call = session.calls[0]
            self.assertIn("MarketLoop", call["system_prompt"])
            user_text = call["messages"][0].content.text
            self.assertIn("Ali Mahmoud", user_text)
            self.assertIn("1", user_text)
            self.assertIn("2026-07-01", user_text)
            self.assertEqual(call["max_tokens"], 300)
            self.assertEqual(call["temperature"], 0.7)
            self.assertEqual(call["related_request_id"], "req-1")

    def test_generate_delay_apology_raises_for_unknown_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "marketloop.db"
            initialize_database(db_path)
            connection = sqlite3.connect(db_path)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON;")
            try:
                with patch("mcp_server.tools.customer_service.get_connection", return_value=connection):
                    with self.assertRaises(ToolValidationError):
                        asyncio.run(
                            generate_delay_apology(
                                {"order_id": 999},
                                context=SimpleNamespace(session=DummySession(), request_id=None),
                            )
                        )
            finally:
                connection.close()

    def test_generate_delay_apology_requires_sampling_capability(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "marketloop.db"
            initialize_database(db_path)
            connection = sqlite3.connect(db_path)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON;")
            try:
                with patch("mcp_server.tools.customer_service.get_connection", return_value=connection):
                    with self.assertRaises(SamplingUnavailableError):
                        asyncio.run(
                            generate_delay_apology(
                                {"order_id": 1},
                                context=SimpleNamespace(session=SimpleNamespace(), request_id=None),
                            )
                        )
            finally:
                connection.close()

    def test_server_call_tool_wires_sampling_context(self) -> None:
        server = MarketLoopMCPServer()
        server.register_modules()
        session = DummySession(text="We are sorry about the delay.")
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "marketloop.db"
            initialize_database(db_path)
            connection = sqlite3.connect(db_path)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON;")
            try:
                with patch("mcp_server.tools.customer_service.get_connection", return_value=connection):
                    result = asyncio.run(
                        server._call_tool(
                            SimpleNamespace(session=session, request_id="req-42"),
                            SimpleNamespace(
                                name="generate_delay_apology",
                                arguments={"order_id": 1},
                                meta=None,
                            ),
                        )
                    )
            finally:
                connection.close()

            self.assertEqual(result.content[0].text, "We are sorry about the delay.")
            self.assertEqual(session.calls[0]["related_request_id"], "req-42")
            self.assertIn("generate_delay_apology", server.list_tools())


if __name__ == "__main__":
    unittest.main()
