"""Configuration helpers for the MarketLoop MCP server."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "marketloop.db"

Transport = Literal["stdio", "streamable-http", "sse"]


def get_transport() -> Transport:
    """Read the transport mode from the environment."""
    value = os.getenv("MARKETLOOP_TRANSPORT", "stdio").strip().lower()
    if value in {"stdio", "streamable-http", "sse"}:
        return value
    return "stdio"


def get_database_path() -> Path:
    """Return the SQLite database path."""
    return DB_PATH


