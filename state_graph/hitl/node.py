from __future__ import annotations

import json
import uuid
from typing import Any

from mcp_server.db import get_connection
from state_graph.core.models import GraphState, TransitionResult


class HITLNode:
    """
    Manage persistent human-intervention requests.

    The node creates a durable HITL request and returns a waiting
    transition so the state-graph engine checkpoints the paused state.
    """

    def create_request(
        self,
        *,
        run_id: str,
        graph_name: str,
        reason: str,
        state: dict[str, Any],
    ) -> str:
        """Create a persistent human-intervention request."""

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

    def pause(
        self,
        state: GraphState,
        *,
        reason: str,
    ) -> TransitionResult:
        """
        Pause the graph for human intervention.

        The exact paused state is created before the request is stored so
        that the human task and the graph checkpoint refer to the same
        state snapshot.
        """

        request_id = str(uuid.uuid4())

        paused_state = state.model_copy(
            update={
                "status": "waiting",
                "waiting_request_id": request_id,
            }
        )

        self._create_request_with_id(
            request_id=request_id,
            run_id=paused_state.run_id,
            graph_name=paused_state.graph_name,
            reason=reason,
            state=paused_state.model_dump(mode="json"),
        )

        return TransitionResult(
            next_node=state.current_node,
            status="waiting",
            updates={
                "waiting_request_id": request_id,
            },
        )

    def _create_request_with_id(
        self,
        *,
        request_id: str,
        run_id: str,
        graph_name: str,
        reason: str,
        state: dict[str, Any],
    ) -> None:
        """Persist a request using an already-created request ID."""

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

    def resolve(
        self,
        request_id: str,
        decision: str,
    ) -> None:
        """Record the administrator's decision."""

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
        """Return one HITL request and its persisted state."""

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