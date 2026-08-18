from __future__ import annotations

import json
from datetime import datetime, timezone

from mcp_server.db import get_connection

from state_graph.core.models import GraphState


class CheckpointStore:
    """
    Persistent checkpoint storage for state graphs.

    A checkpoint contains the complete serializable GraphState,
    allowing a graph to resume after process restart.
    """

    def __init__(self) -> None:
        self._ensure_table()

    def _ensure_table(self) -> None:
        with get_connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS State_Checkpoints (
                    checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    graph_name TEXT NOT NULL,
                    current_node TEXT NOT NULL,
                    status TEXT NOT NULL,
                    transition_count INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_state_checkpoints_run
                ON State_Checkpoints(run_id, checkpoint_id)
                """
            )

            connection.commit()

    def save(self, state: GraphState) -> None:
        state_json = json.dumps(
            state.model_dump(mode="json"),
            sort_keys=True,
        )

        created_at = datetime.now(
            timezone.utc
        ).isoformat()

        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO State_Checkpoints (
                    run_id,
                    graph_name,
                    current_node,
                    status,
                    transition_count,
                    state_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    state.run_id,
                    state.graph_name,
                    state.current_node,
                    state.status,
                    state.transition_count,
                    state_json,
                    created_at,
                ),
            )

            connection.commit()

    def load_latest(
        self,
        run_id: str,
    ) -> GraphState | None:

        with get_connection() as connection:
            row = connection.execute(
                """
                SELECT state_json
                FROM State_Checkpoints
                WHERE run_id = ?
                ORDER BY checkpoint_id DESC
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()

        if row is None:
            return None

        state_json = row[0]

        return GraphState.model_validate(
            json.loads(state_json)
        )