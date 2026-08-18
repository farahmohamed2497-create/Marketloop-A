from __future__ import annotations

import json
import uuid
from typing import Any

from mcp_server.db import get_connection


class HITLNode:
    """
    Persistent human-intervention request manager.
    """

    def create_request(
        self,
        *,
        run_id: str,
        graph_name: str,
        reason: str,
        state: dict[str, Any],
    ) -> str:

        request_id = str(uuid.uuid4())

        with get_connection() as connection:

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS HITL_Requests (
                    request_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    graph_name TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    state TEXT NOT NULL,
                    decision TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMP
                        DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TIMESTAMP
                )
                """
            )

            connection.execute(
                """
                INSERT INTO HITL_Requests (
                    request_id,
                    run_id,
                    graph_name,
                    reason,
                    state
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    run_id,
                    graph_name,
                    reason,
                    json.dumps(
                        state,
                        sort_keys=True,
                    ),
                ),
            )

            connection.commit()

        return request_id

    def resolve(
        self,
        request_id: str,
        decision: str,
    ) -> None:

        normalized = decision.strip().lower()

        if normalized not in {
            "approve",
            "reject",
        }:
            raise ValueError(
                "HITL decision must be "
                "'approve' or 'reject'"
            )

        with get_connection() as connection:

            connection.execute(
                """
                UPDATE HITL_Requests
                SET decision = ?,
                    status = 'resolved',
                    resolved_at = CURRENT_TIMESTAMP
                WHERE request_id = ?
                """,
                (
                    normalized,
                    request_id,
                ),
            )

            connection.commit()

    def get_request(
        self,
        request_id: str,
    ) -> dict[str, Any] | None:

        with get_connection() as connection:

            row = connection.execute(
                """
                SELECT
                    request_id,
                    run_id,
                    graph_name,
                    reason,
                    state,
                    decision,
                    status
                FROM HITL_Requests
                WHERE request_id = ?
                """,
                (request_id,),
            ).fetchone()

        if row is None:
            return None

        return {
            "request_id": row[0],
            "run_id": row[1],
            "graph_name": row[2],
            "reason": row[3],
            "state": json.loads(row[4]),
            "decision": row[5],
            "status": row[6],
        }