from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

import mcp_server.db as mcp_db
import admin_api.main as admin_main


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Point every mcp_server.db.get_connection() call at a throwaway
    sqlite file for the duration of one test.

    get_connection() re-resolves get_database_path() on every call (it's
    not cached), so patching the name in mcp_server.db's own namespace
    redirects RagDocumentStore / HITLNode / FailureTicketService /
    agent_registry alike, however/whenever they were constructed.
    """
    db_path = tmp_path / "test_marketloop.db"
    monkeypatch.setattr(mcp_db, "get_database_path", lambda: db_path)

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS Agents (
                agent_id TEXT PRIMARY KEY,
                display_name TEXT,
                description TEXT,
                agent_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen_at TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS Agent_Tools (
                agent_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                updated_at TIMESTAMP,
                updated_by TEXT,
                PRIMARY KEY (agent_id, tool_name)
            )
            """
        )
        connection.commit()

    return db_path


@pytest.fixture
def client(isolated_db, monkeypatch):
    """A TestClient wired to the admin API with its lazy singletons reset
    so they get rebuilt against `isolated_db` on first use."""
    monkeypatch.setattr(admin_main, "rag_documents", None)
    monkeypatch.setattr(admin_main, "hitl", None)
    monkeypatch.setattr(admin_main, "failure_tickets", None)

    return TestClient(admin_main.app)