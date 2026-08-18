from __future__ import annotations

from state_graph.core.exceptions import CheckpointNotFoundError
from state_graph.core.models import GraphState

from .store import CheckpointStore


def recover_run(
    run_id: str,
    store: CheckpointStore | None = None,
) -> GraphState:
    """Recover the latest persisted state for a run."""

    store = store or CheckpointStore()

    state = store.load_latest(run_id)

    if state is None:
        raise CheckpointNotFoundError(
            f"No checkpoint found for run_id={run_id!r}"
        )

    return state