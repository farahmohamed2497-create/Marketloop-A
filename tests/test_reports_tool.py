import asyncio
import json
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
from mcp_server.tools.reports import (
    ToolValidationError,
    generate_sales_audit_report,
)


class DummySession:
    def __init__(self) -> None:
        self.progress_updates: list[dict[str, object]] = []

    async def send_progress_notification(
        self,
        progress_token: str | int,
        progress: float,
        total: float | None = None,
        message: str | None = None,
        related_request_id: str | None = None,
    ) -> None:
        self.progress_updates.append(
            {
                "progress_token": progress_token,
                "progress": progress,
                "total": total,
                "message": message,
            }
        )


class ReportToolTests(unittest.TestCase):
    def _build_connection(self, temp_dir: str) -> tuple[object, object]:
        db_path = Path(temp_dir) / "marketloop.db"
        initialize_database(db_path)
        connection = sqlite3.connect(db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON;")
        return connection, db_path

    def test_generate_sales_audit_report_sends_progress_notifications(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            connection, _ = self._build_connection(temp_dir)
            session = DummySession()
            try:
                with patch("mcp_server.tools.reports.get_connection", return_value=connection):
                    asyncio.run(
                        generate_sales_audit_report(
                            {"start_date": "2026-07-01", "end_date": "2026-07-31"},
                            context=SimpleNamespace(session=session, progress_token="tok-1"),
                        )
                    )
            finally:
                connection.close()

            expected = [(0.00, "Initializing report generation..."),
                        (0.25, "Aggregating total sales revenue..."),
                        (0.50, "Calculating return ratios and refund metrics..."),
                        (0.75, "Checking inventory levels and system audit logs..."),
                        (1.00, "Report complete.")]
            self.assertEqual(
                [(update["progress"], update["message"]) for update in session.progress_updates],
                expected,
            )
            self.assertEqual(session.progress_updates[0]["progress_token"], "tok-1")
            self.assertEqual(session.progress_updates[0]["total"], 1.0)

    def test_generate_sales_audit_report_aggregates_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            connection, _ = self._build_connection(temp_dir)
            try:
                with patch("mcp_server.tools.reports.get_connection", return_value=connection):
                    result = asyncio.run(
                        generate_sales_audit_report(
                            {"start_date": "2026-07-01", "end_date": "2026-07-31"},
                            context=SimpleNamespace(session=None, progress_token=None),
                        )
                    )
            finally:
                connection.close()

            report = json.loads(result)
            self.assertEqual(report["report_type"], "sales_audit_report")
            self.assertEqual(report["period"], {"start_date": "2026-07-01", "end_date": "2026-07-31"})
            self.assertEqual(report["sales"]["total_orders"], 3)
            self.assertEqual(report["sales"]["total_revenue"], 55700.0)
            self.assertEqual(report["sales"]["units_sold"], 4)
            self.assertEqual(report["returns"]["total_returns"], 1)
            self.assertEqual(report["returns"]["return_rate"], round(1 / 3, 4))
            self.assertIn("low_stock_items", report["inventory"])
            self.assertIn("audit_events", report["inventory"])

    def test_generate_sales_audit_report_honors_date_window(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            connection, _ = self._build_connection(temp_dir)
            try:
                with patch("mcp_server.tools.reports.get_connection", return_value=connection):
                    result = asyncio.run(
                        generate_sales_audit_report(
                            {"start_date": "2026-07-06", "end_date": "2026-07-31"},
                            context=SimpleNamespace(session=None, progress_token=None),
                        )
                    )
            finally:
                connection.close()

            report = json.loads(result)
            self.assertEqual(report["sales"]["total_orders"], 1)
            self.assertEqual(report["sales"]["total_revenue"], 2700.0)
            self.assertEqual(report["returns"]["total_returns"], 1)

    def test_generate_sales_audit_report_rejects_invalid_dates(self) -> None:
        with self.assertRaises(ToolValidationError):
            asyncio.run(
                generate_sales_audit_report(
                    {"start_date": "2026-07-31", "end_date": "2026-07-01"},
                    context=SimpleNamespace(session=None, progress_token=None),
                )
            )
        with self.assertRaises(ToolValidationError):
            asyncio.run(
                generate_sales_audit_report(
                    {"start_date": "07/01/2026", "end_date": "2026-07-31"},
                    context=SimpleNamespace(session=None, progress_token=None),
                )
            )

    def test_server_call_tool_wires_progress_token(self) -> None:
        server = MarketLoopMCPServer()
        server.register_modules()
        session = DummySession()
        with tempfile.TemporaryDirectory() as temp_dir:
            connection, _ = self._build_connection(temp_dir)
            try:
                with patch("mcp_server.tools.reports.get_connection", return_value=connection):
                    result = asyncio.run(
                        server._call_tool(
                            SimpleNamespace(session=session),
                            SimpleNamespace(
                                name="generate_sales_audit_report",
                                arguments={"start_date": "2026-07-01", "end_date": "2026-07-31"},
                                meta={"progressToken": "tok-abc"},
                            ),
                        )
                    )
            finally:
                connection.close()

            self.assertEqual(session.progress_updates[0]["progress_token"], "tok-abc")
            self.assertEqual(
                [update["progress"] for update in session.progress_updates],
                [0.00, 0.25, 0.50, 0.75, 1.00],
            )
            self.assertIn("total_revenue", result.content[0].text)
            self.assertIn("generate_sales_audit_report", server.list_tools())


if __name__ == "__main__":
    unittest.main()
