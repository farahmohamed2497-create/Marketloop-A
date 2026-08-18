from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from mcp_server.db import get_connection


class FailureTicketService:
    """
    Persistent failure-ticket service.

    A failure ticket records the node failure and the state
    that existed when the failure happened.
    """

    def __init__(self) -> None:
        self._ensure_table()

    def _ensure_table(self) -> None:
        with get_connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS Failure_Tickets (
                    ticket_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    graph_name TEXT NOT NULL,
                    node_name TEXT NOT NULL,
                    error TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    state_json TEXT,
                    created_at TEXT NOT NULL,
                    resolved_at TEXT
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_failure_tickets_run
                ON Failure_Tickets(run_id)
                """
            )

            connection.commit()

    def create_ticket(
        self,
        *,
        run_id: str,
        graph_name: str,
        node_name: str,
        error: str,
        state: dict[str, Any] | None = None,
    ) -> str:

        ticket_id = str(uuid.uuid4())

        created_at = datetime.now(
            timezone.utc
        ).isoformat()

        state_json = (
            json.dumps(
                state,
                sort_keys=True,
            )
            if state is not None
            else None
        )

        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO Failure_Tickets (
                    ticket_id,
                    run_id,
                    graph_name,
                    node_name,
                    error,
                    status,
                    state_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, 'open', ?, ?)
                """,
                (
                    ticket_id,
                    run_id,
                    graph_name,
                    node_name,
                    error,
                    state_json,
                    created_at,
                ),
            )

            connection.commit()

        return ticket_id

    def resolve_ticket(
        self,
        ticket_id: str,
    ) -> None:

        resolved_at = datetime.now(
            timezone.utc
        ).isoformat()

        with get_connection() as connection:
            connection.execute(
                """
                UPDATE Failure_Tickets
                SET status = 'resolved',
                    resolved_at = ?
                WHERE ticket_id = ?
                """,
                (
                    resolved_at,
                    ticket_id,
                ),
            )

            connection.commit()