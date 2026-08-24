from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Callable
from typing import Any

from mcp_server.db import get_connection


ConnectionFactory = Callable[[], sqlite3.Connection]


class ChatSessionStore:
    """Persist F3 conversations without sharing state across agents."""

    def __init__(self, connection_factory: ConnectionFactory = get_connection) -> None:
        self._connection_factory = connection_factory
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._connection_factory() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS Platform_Chat_Sessions (
                    session_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS Platform_Chat_Messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES Platform_Chat_Sessions(session_id)
                )"""
            )
            connection.commit()

    def get_or_create(self, *, agent_id: str, session_id: str | None = None) -> str:
        with self._connection_factory() as connection:
            if session_id:
                row = connection.execute(
                    "SELECT agent_id FROM Platform_Chat_Sessions WHERE session_id = ?", (session_id,)
                ).fetchone()
                if row is None:
                    raise ValueError("Unknown chat session.")
                if row[0] != agent_id:
                    raise ValueError("A chat session cannot be used with a different agent.")
                return session_id
            created = str(uuid.uuid4())
            connection.execute(
                "INSERT INTO Platform_Chat_Sessions (session_id, agent_id) VALUES (?, ?)",
                (created, agent_id),
            )
            connection.commit()
            return created

    def add_message(self, *, session_id: str, role: str, content: str) -> None:
        with self._connection_factory() as connection:
            connection.execute(
                "INSERT INTO Platform_Chat_Messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content),
            )
            connection.commit()

    def history(self, session_id: str) -> list[dict[str, Any]]:
        with self._connection_factory() as connection:
            rows = connection.execute(
                "SELECT role, content, created_at FROM Platform_Chat_Messages WHERE session_id = ? ORDER BY message_id",
                (session_id,),
            ).fetchall()
        return [{"role": row[0], "content": row[1], "created_at": row[2]} for row in rows]
