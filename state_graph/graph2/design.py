"""Explicit state design for the shipping-investigation graph."""

from __future__ import annotations

from enum import Enum

from state_graph.core.transitions import TransitionTable


class ShippingState(str, Enum):
    """Named states for a shipping issue that can span carrier responses."""

    AWAITING_INPUT = "awaiting_input"
    DECOMPOSE = "decompose"
    CONSTRAINED_REACT = "constrained_react"
    AWAITING_CARRIER = "awaiting_carrier"
    DONE = "done"


def shipping_transition_table() -> TransitionTable:
    """Build the allowed transitions, including the carrier-response cycle."""

    transitions = TransitionTable()
    transitions.add(ShippingState.AWAITING_INPUT.value, ShippingState.DECOMPOSE.value)
    transitions.add(ShippingState.DECOMPOSE.value, ShippingState.CONSTRAINED_REACT.value)
    transitions.add_many(
        ShippingState.CONSTRAINED_REACT.value,
        [
            ShippingState.AWAITING_CARRIER.value,
            ShippingState.DONE.value,
        ],
    )
    # A carrier response changes the available evidence, so the graph loops
    # back through the constrained decision node instead of starting over.
    transitions.add(
        ShippingState.AWAITING_CARRIER.value,
        ShippingState.CONSTRAINED_REACT.value,
    )
    return transitions
