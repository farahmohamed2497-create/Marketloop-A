from __future__ import annotations

from state_graph.core.exceptions import CheckpointNotFoundError
from state_graph.core.models import GraphState

from state_graph.checkpointing.store import CheckpointStore
from state_graph.failure_tickets.service import FailureTicketService
from state_graph.tickets.service import FailureTicketService

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


def resume_after_failure(
    run_id: str,
    *,
    store: CheckpointStore | None = None,
    failure_tickets: FailureTicketService | None = None,
) -> GraphState:
    """
    Resume a failed Graph 2 run only after its failure ticket
    has been resolved.

    The graph resumes from the persisted failed node rather than
    starting again from awaiting_input.
    """

    store = store or CheckpointStore()
    failure_tickets = (
        failure_tickets or FailureTicketService()
    )

    state = recover_run(
        run_id,
        store=store,
    )

    ticket_id = state.data.get("failure_ticket_id")

    if not ticket_id:
        raise ValueError(
            f"Run {run_id!r} has no failure ticket."
        )

    ticket = failure_tickets.get_ticket(ticket_id)

    if ticket is None:
        raise ValueError(
            f"Failure ticket {ticket_id!r} was not found."
        )

    if ticket["status"] != "resolved":
        raise RuntimeError(
            f"Failure ticket {ticket_id!r} is still "
            f"{ticket['status']!r}."
        )

    resumed_data = {
        **state.data,
        "failure_resolved": True,
    }

    resumed_state = state.model_copy(
        update={
            "status": "running",
            "data": resumed_data,
        }
    )

    store.save(resumed_state)

    return resumed_state