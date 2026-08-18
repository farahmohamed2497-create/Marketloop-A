from __future__ import annotations

from collections.abc import Callable

from .exceptions import InvalidTransitionError
from .models import GraphState, TransitionResult
from .transitions import TransitionTable

from state_graph.checkpointing.store import CheckpointStore
from state_graph.tickets.service import FailureTicketService


NodeFunction = Callable[
    [GraphState],
    TransitionResult,
]


class StateGraphEngine:
    """
    Execute a persistent state graph node by node.
    """

    def __init__(
        self,
        *,
        transitions: TransitionTable,
        nodes: dict[str, NodeFunction],
        checkpoint_store: CheckpointStore | None = None,
        ticket_service: FailureTicketService | None = None,
    ) -> None:

        self.transitions = transitions
        self.nodes = nodes

        self.checkpoint_store = (
            checkpoint_store
            or CheckpointStore()
        )

        self.ticket_service = (
            ticket_service
            or FailureTicketService()
        )

    def initialize(
        self,
        state: GraphState,
    ) -> GraphState:

        self.checkpoint_store.save(state)

        return state

    def step(
        self,
        state: GraphState,
    ) -> GraphState:

        node = state.current_node

        if node not in self.nodes:
            raise InvalidTransitionError(
                f"Unknown graph node: {node}"
            )

        try:
            result = self.nodes[node](state)

            if result.next_node != node:
                self.transitions.validate(
                    node,
                    result.next_node,
                )

            updated = state.model_copy(
                update={
                    **result.updates,

                    "current_node":
                        result.next_node,

                    "transition_count":
                        state.transition_count + 1,

                    "status":
                        (
                            result.status
                            if result.status is not None
                            else state.status
                        ),

                    "last_error":
                        result.error,
                }
            )

            self.checkpoint_store.save(
                updated
            )

            return updated

        except Exception as exc:

            ticket_id = (
                self.ticket_service.create_ticket(
                    run_id=state.run_id,
                    graph_name=state.graph_name,
                    node_name=node,
                    error=str(exc),
                    state=state.model_dump(
                        mode="json"
                    ),
                )
            )

            failed = state.model_copy(
                update={
                    "status": "failed",

                    "last_error":
                        str(exc),

                    "waiting_ticket_id":
                        ticket_id,
                }
            )

            self.checkpoint_store.save(
                failed
            )

            return failed

    def run(
        self,
        state: GraphState,
        *,
        max_steps: int = 100,
    ) -> GraphState:

        if state.transition_count == 0:
            self.initialize(state)

        current = state

        for _ in range(max_steps):

            if current.status in {
                "done",
                "failed",
                "waiting",
            }:
                return current

            current = self.step(current)

        failed = current.model_copy(
            update={
                "status": "failed",

                "last_error":
                    "Maximum state-graph transitions exceeded.",
            }
        )

        self.checkpoint_store.save(
            failed
        )

        return failed

    def recover(
        self,
        run_id: str,
    ) -> GraphState:

        state = (
            self.checkpoint_store
            .load_latest(run_id)
        )

        if state is None:
            raise ValueError(
                f"No checkpoint found for "
                f"run_id={run_id!r}"
            )

        return state

    def resume(
            self,
            run_id: str,
    ) -> GraphState:
        """Resume a waiting or failed run from its latest checkpoint."""

        state = self.recover(run_id)

        if state.status not in {"waiting", "failed"}:
            raise ValueError(
                f"Run {run_id!r} cannot be resumed "
                f"from status={state.status!r}"
            )

        if state.status == "waiting":
            state = state.model_copy(
                update={
                    "status": "running",
                    "waiting_request_id": None,
                }
            )

        elif state.status == "failed":
            state = state.model_copy(
                update={
                    "status": "running",
                    "last_error": None,
                    "waiting_ticket_id": None,
                }
            )

        self.checkpoint_store.save(state)

        return self.run(state)


