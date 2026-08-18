from __future__ import annotations

from .models import FailureTicket
from mcp_server.db import get_connection


class FailureTicketRepository:
    """Database persistence for failure tickets."""

    def initialize_schema(self) -> None:
        with get_connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS Failure_Tickets (
                    ticket_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    graph_name TEXT NOT NULL,
                    node_name TEXT NOT NULL,
                    error TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    resolution TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_failure_ticket_run
                ON Failure_Tickets(run_id);

                CREATE INDEX IF NOT EXISTS idx_failure_ticket_status
                ON Failure_Tickets(status);
                """
            )
            connection.commit()

    def create(self, ticket: FailureTicket) -> int:
        self.initialize_schema()

        with get_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO Failure_Tickets (
                    run_id,
                    graph_name,
                    node_name,
                    error,
                    status
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    ticket.run_id,
                    ticket.graph_name,
                    ticket.node_name,
                    ticket.error,
                    ticket.status,
                ),
            )

            connection.commit()

            return int(cursor.lastrowid)

    def resolve(
        self,
        ticket_id: int,
        resolution: str,
    ) -> None:
        self.initialize_schema()

        with get_connection() as connection:
            connection.execute(
                """
                UPDATE Failure_Tickets
                SET status = 'resolved',
                    resolution = ?,
                    resolved_at = CURRENT_TIMESTAMP
                WHERE ticket_id = ?
                """,
                (resolution, ticket_id),
            )

            connection.commit()

    def get(self, ticket_id: int) -> FailureTicket | None:
        self.initialize_schema()

        with get_connection() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM Failure_Tickets
                WHERE ticket_id = ?
                """,
                (ticket_id,),
            ).fetchone()

        if row is None:
            return None

        return FailureTicket(
            ticket_id=row["ticket_id"],
            run_id=row["run_id"],
            graph_name=row["graph_name"],
            node_name=row["node_name"],
            error=row["error"],
            status=row["status"],
            resolution=row["resolution"],
        )