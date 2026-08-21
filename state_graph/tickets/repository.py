from __future__ import annotations

from .models import FailureTicket
from .service import FailureTicketService


class FailureTicketRepository:
    """Compatibility facade over the canonical failure-ticket service."""

    def __init__(self, service: FailureTicketService | None = None) -> None:
        self.service = service or FailureTicketService()

    def initialize_schema(self) -> None:
        """Kept for callers of the original repository API."""

    def create(self, ticket: FailureTicket) -> str:
        return self.service.create_ticket(
            run_id=ticket.run_id,
            graph_name=ticket.graph_name,
            node_name=ticket.node_name,
            error=ticket.error,
        )

    def resolve(
        self,
        ticket_id: str,
        resolution: str,
    ) -> None:
        self.service.resolve_ticket(ticket_id, resolution)

    def get(self, ticket_id: str) -> FailureTicket | None:
        ticket = self.service.get_ticket(ticket_id)

        if ticket is None:
            return None

        return FailureTicket(
            ticket_id=ticket["ticket_id"],
            run_id=ticket["run_id"],
            graph_name=ticket["graph_name"],
            node_name=ticket["node_name"],
            error=ticket["error"],
            status=ticket["status"],
        )
