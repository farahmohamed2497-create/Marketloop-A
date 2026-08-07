from mcp_server.memory.rolling_buffer import RollingBuffer
from mcp_server.memory.promote_drop_router import PromoteDropRouter


def test_memory_flow_end_to_end():

    buffer = RollingBuffer(max_turns=2)

    router = PromoteDropRouter()

    evicted = buffer.add_turn(
        "user",
        "return reason: item arrived damaged"
    )

    assert evicted is None

    buffer.add_turn(
        "tool",
        "lookup result"
    )

    evicted = buffer.add_turn(
        "tool",
        "shipment result"
    )

    assert evicted is not None

    router.route([evicted])

    assert len(router.episodic_store) == 1