from state_graph.core.models import GraphState
from state_graph.graph2.design import ShippingState, shipping_transition_table
from state_graph.graph2.shipping_graph import ShippingGraph


def test_shipping_graph_declares_states_and_carrier_cycle():
    transitions = shipping_transition_table()

    assert ShippingState.AWAITING_CARRIER.value == "awaiting_carrier"
    assert transitions.allowed("constrained_react", "awaiting_carrier")
    assert transitions.allowed("awaiting_carrier", "constrained_react")
    assert transitions.allowed("constrained_react", "done")


def test_awaiting_carrier_waits_then_returns_to_react():
    graph = ShippingGraph(llm=object())
    waiting = graph.awaiting_carrier(
        GraphState(
            run_id="shipping-cycle",
            graph_name="shipping",
            current_node="awaiting_carrier",
        )
    )
    resumed = graph.awaiting_carrier(
        GraphState(
            run_id="shipping-cycle",
            graph_name="shipping",
            current_node="awaiting_carrier",
            data={"carrier_response": {"status": "investigating"}},
        )
    )

    assert waiting.status == "waiting"
    assert waiting.next_node == "awaiting_carrier"
    assert resumed.next_node == "constrained_react"
