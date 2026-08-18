from __future__ import annotations

import sqlite3
from typing import Any
from .store import CheckpointStore


repository = CheckpointStore()

__all__ = ["CheckpointStore", "repository"]


class CheckpointRepository:
    """
    SQLite persistence layer for durable graph checkpoints.

    The default database is state_graph.db, but callers can provide
    the project's existing DB path.
    """

    def __init__(self, db_path: str = "state_graph.db") -> None:
        self.db_path = db_path
        self._ensure_schema()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS
                state_graph_checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    graph_name TEXT NOT NULL,
                    node TEXT NOT NULL,
                    status TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_checkpoint_run
                ON state_graph_checkpoints(run_id, version)
                """
            )

    def save(
        self,
        *,
        run_id: str,
        graph_name: str,
        node: str,
        status: str,
        version: int,
        state_json: str,
        created_at: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO state_graph_checkpoints
                (
                    run_id,
                    graph_name,
                    node,
                    status,
                    version,
                    state_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    graph_name,
                    node,
                    status,
                    version,
                    state_json,
                    created_at,
                ),
            )

    def latest(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT
                    run_id,
                    graph_name,
                    node,
                    status,
                    version,
                    state_json,
                    created_at
                FROM state_graph_checkpoints
                WHERE run_id = ?
                ORDER BY version DESC
                LIMIT 1
                """,
                (run_id,),
            )

            row = cursor.fetchone()

        if row is None:
            return None

        return {
            "run_id": row[0],
            "graph_name": row[1],
            "node": row[2],
            "status": row[3],
            "version": row[4],
            "state_json": row[5],
            "created_at": row[6],
        }