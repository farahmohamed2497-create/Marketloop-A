from __future__ import annotations

import sqlite3

from .config import get_database_path


def get_connection() -> sqlite3.Connection:
    db_path = get_database_path()
    db_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        db_path,
    )

    connection.row_factory = sqlite3.Row

    connection.execute(
        "PRAGMA foreign_keys = ON;"
    )

    return connection


def close_connection(
    connection: sqlite3.Connection,
) -> None:
    if connection:
        connection.close()


class DatabaseSession:
    def __init__(self) -> None:
        self.connection: sqlite3.Connection | None = None

    def __enter__(self) -> sqlite3.Connection:
        self.connection = get_connection()
        return self.connection

    def __exit__(
        self,
        exc_type,
        exc,
        tb,
    ) -> None:
        if self.connection is not None:
            self.connection.close()
