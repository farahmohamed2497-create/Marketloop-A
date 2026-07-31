"""Initialize the MarketLoop SQLite database from SQL scripts."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def initialize_database(db_path: str | Path | None = None) -> Path:
    """Create or reset the MarketLoop database from schema and seed SQL files."""
    base_dir = Path(__file__).resolve().parent.parent
    db_file = Path(db_path) if db_path is not None else base_dir / "marketloop.db"
    schema_path = base_dir / "db" / "schema.sql"
    seed_path = base_dir / "db" / "seed_data.sql"

    db_file.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_file) as connection:
        connection.execute("PRAGMA foreign_keys = ON;")
        connection.executescript(schema_path.read_text(encoding="utf-8"))
        connection.executescript(seed_path.read_text(encoding="utf-8"))
        connection.commit()

    return db_file


if __name__ == "__main__":
    initialize_database()
