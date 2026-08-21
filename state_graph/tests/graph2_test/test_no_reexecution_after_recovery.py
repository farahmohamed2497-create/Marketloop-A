from __future__ import annotations

from unittest.mock import Mock

from state_graph.core.engine import StateGraphEngine
from state_graph.core.models import GraphState
from state_graph.checkpointing.store import CheckpointStore
from state_graph.tickets.service import FailureTicketService

from state_graph.graph2.graph import build_graph2


def test_shipping_graph_creates_recovery_ticket_on_tracking_failure():
    llm = Mock()

    checkpoint_store = CheckpointStore()
    ticket_service = FailureTicketService()

    engine = build_graph2(
        llm=llm,
        checkpoint_store=checkpoint_store,
        ticket_service=ticket_service,
    )

    state = GraphState(
        run_id="shipping-graph-failure-test",
        graph_name="shipping",
        current_node="tracking",
        goal=(
            "Customer reports that the package did not arrive "
            "and carrier tracking data is contradictory."
        ),
    )

    result = engine.step(state)

    assert result.status == "failed"
    assert result.waiting_ticket_id is not None

    ticket = ticket_service.get_ticket(
        result.waiting_ticket_id
    )

    assert ticket is not None
    assert ticket["graph_name"] == "shipping"
    assert ticket["status"] == "open"