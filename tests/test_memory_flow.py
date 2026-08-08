from mcp_server.memory.rolling_buffer import RollingBuffer
from mcp_server.memory.promote_drop_router import PromoteDropRouter
from mcp_server.memory.semantic_memory import SemanticMemory
from mcp_server.memory.consolidation import ConsolidationLayer

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

    import json
    from types import SimpleNamespace

    class FakeMessages:
        @staticmethod
        def create(model, max_tokens, system, messages):
            payload = json.dumps({
                "entity": "customer_1",
                "fact_type": "return_request",
                "value": messages[0]["content"]
            })
            return SimpleNamespace(
                content=[SimpleNamespace(text=payload)]
            )

    class FakeAnthropicClient:
        def __init__(self):
            self.messages = FakeMessages()


    semantic = SemanticMemory()

    consolidation = ConsolidationLayer(
        episodic_store=router.episodic_store,
        semantic_memory=semantic,
        llm_client=FakeAnthropicClient(),
    )

    consolidation.run()

    assert len(semantic.get_all_current()) >= 1
