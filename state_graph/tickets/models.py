from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FailureTicket(BaseModel):
    """Persistent representation of an execution failure."""

    model_config = ConfigDict(extra="forbid")

    ticket_id: int | None = None

    run_id: str = Field(min_length=1)

    graph_name: str = Field(min_length=1)

    node_name: str = Field(min_length=1)

    error: str = Field(min_length=1)

    status: str = "open"

    resolution: str | None = None