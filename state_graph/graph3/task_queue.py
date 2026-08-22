"""Admin task-queue adapter used by Graph 3 HITL pauses."""

from __future__ import annotations

import json
from typing import Any, Protocol

from mcp_server.db import get_connection


class AdminTaskQueue(Protocol):
    def enqueue_hitl(self, *, request_id: str, run_id: str, graph_name: str, state: dict[str, Any]) -> None: ...


class DatabaseAdminTaskQueue:
    """Durable queue the platform can poll to show pending admin work."""

    def enqueue_hitl(self, *, request_id: str, run_id: str, graph_name: str, state: dict[str, Any]) -> None:
        with get_connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS Admin_Task_Queue (
                    request_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    graph_name TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    state_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO Admin_Task_Queue
                (request_id, run_id, graph_name, status, state_json)
                VALUES (?, ?, ?, 'pending', ?)
                """,
                (request_id, run_id, graph_name, json.dumps(state, sort_keys=True)),
            )
            connection.commit()
