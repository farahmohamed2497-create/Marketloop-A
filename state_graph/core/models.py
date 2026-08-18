from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GraphStatus(str, Enum):
    """Possible lifecycle states of a state-graph run."""

    RUNNING = "running"
    WAITING = "waiting"
    DONE = "done"
    FAILED = "failed"


class GraphState(BaseModel):
    """Serializable state persisted between graph transitions."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    graph_name: str = Field(min_length=1)

    current_node: str = Field(min_length=1)

    status: str = GraphStatus.RUNNING.value

    goal: str = ""

    data: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)

    transition_count: int = Field(default=0, ge=0)

    last_error: str | None = None

    waiting_request_id: str | None = None
    waiting_ticket_id: str | None = None


class TransitionResult(BaseModel):
    """Result produced by a state-graph node."""

    model_config = ConfigDict(extra="forbid")

    next_node: str

    updates: dict[str, Any] = Field(default_factory=dict)

    status: str | None = None

    error: str | None = None