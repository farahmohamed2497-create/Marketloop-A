from __future__ import annotations

from state_graph.core.engine import StateGraphEngine
from state_graph.core.models import (
    GraphState,
    TransitionResult,
)
from state_graph.tickets.service import (
    FailureTicketService,
)


class RecoveryGraph:
    """Graph 3: checkpoint, failure ticket and resume."""

    def __init__(
        self,
        engine: StateGraphEngine,
    ) -> None:
        self.engine = engine
        self.tickets = FailureTicketService()

    def execute(
        self,
        state: GraphState,
    ) -> TransitionResult:
        return TransitionResult(
            next_node="checkpoint",
        )

    def checkpoint(
        self,
        state: GraphState,
    ) -> TransitionResult:
        self.engine.checkpoint_store.save(
            state
        )

        return TransitionResult(
            next_node="execute",
        )

    def classify_failure(
        self,
        state: GraphState,
    ) -> TransitionResult:
        if state.status != "failed":
            return TransitionResult(
                next_node="done",
                status="done",
            )

        return TransitionResult(
            next_node="ticket",
        )

    def ticket(
        self,
        state: GraphState,
    ) -> TransitionResult:
        ticket_id = self.tickets.create_ticket(
            run_id=state.run_id,
            graph_name=state.graph_name,
            node_name=state.current_node,
            error=state.last_error
            or "Unknown execution failure",
        )

        return TransitionResult(
            next_node="waiting",
            status="waiting",
            updates={
                "waiting_ticket_id": str(
                    ticket_id
                )
            },
        )

    def resume(
        self,
        state: GraphState,
    ) -> TransitionResult:
        if not state.waiting_ticket_id:
            raise ValueError(
                "No failure ticket attached to state."
            )

        return TransitionResult(
            next_node="execute",
            status="running",
            updates={
                "waiting_ticket_id": None,
                "last_error": None,
            },
        )

    def nodes(self):
        return {
            "execute": self.execute,
            "checkpoint": self.checkpoint,
            "classify_failure":
                self.classify_failure,
            "ticket": self.ticket,
            "resume": self.resume,
        }