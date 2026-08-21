from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from mcp_server.db import get_connection


class FailureTicketService:
    """
    Persistent failure-ticket service.

    A failure ticket records the node failure and the state
    that existed when the failure happened.
    """

    def __init__(
        self,
        connection_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._get_connection = connection_factory or get_connection
        self._ensure_table()

    def _ensure_table(self) -> None:
        with self._get_connection() as connection:
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

        with self._get_connection() as connection:
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
        resolution: str | None = None,
    ) -> None:

        resolved_at = datetime.now(
            timezone.utc
        ).isoformat()

        with self._get_connection() as connection:
            cursor = connection.execute(
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

            if cursor.rowcount != 1:
                raise ValueError(f"Unknown failure ticket: {ticket_id}")

            connection.commit()

    def get_ticket(self, ticket_id: str) -> dict[str, Any] | None:
        """Return the persisted failure context needed by an administrator."""

        with self._get_connection() as connection:
            row = connection.execute(
                """
                SELECT ticket_id, run_id, graph_name, node_name, error,
                       status, state_json, created_at, resolved_at
                FROM Failure_Tickets
                WHERE ticket_id = ?
                """,
                (ticket_id,),
            ).fetchone()

        if row is None:
            return None

        return {
            "ticket_id": row[0],
            "run_id": row[1],
            "graph_name": row[2],
            "node_name": row[3],
            "error": row[4],
            "status": row[5],
            "state": json.loads(row[6]) if row[6] else None,
            "created_at": row[7],
            "resolved_at": row[8],
        }
