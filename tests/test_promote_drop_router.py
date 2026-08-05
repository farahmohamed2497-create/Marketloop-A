import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from mcp_server.memory.promote_drop_router import PromoteDropRouter


def test_tool_output_is_forgotten():
    router = PromoteDropRouter()
    decisions = router.route([
        {"role": "tool", "content": "[get_order_details(order_id=4521)] -> OK"}
    ])

    assert decisions[0].decision == "forget"
    assert len(router.episodic_store) == 0


def test_return_reason_is_promoted():
    router = PromoteDropRouter()
    decisions = router.route([
        {"role": "user", "content": "Return reason: item arrived damaged in shipping"}
    ])

    assert decisions[0].decision == "promote"
    assert len(router.episodic_store) == 1
    assert "damaged" in router.episodic_store[0].content


def test_routine_chat_is_forgotten():
    router = PromoteDropRouter()
    decisions = router.route([
        {"role": "user", "content": "ok thanks, sounds good"}
    ])

    assert decisions[0].decision == "forget"


def test_address_change_is_promoted():
    router = PromoteDropRouter()
    decisions = router.route([
        {"role": "user", "content": "please update my shipping address for future orders"}
    ])

    assert decisions[0].decision == "promote"


def test_system_summary_is_forgotten():
    router = PromoteDropRouter()
    decisions = router.route([
        {"role": "system", "content": "[Summary of 10 earlier turns] ..."}
    ])

    assert decisions[0].decision == "forget"


def test_every_decision_has_logged_reasoning():
    router = PromoteDropRouter()
    router.route([
        {"role": "tool", "content": "noise"},
        {"role": "user", "content": "customer complaint about damaged laptop"},
    ])

    log = router.get_reasoning_log()

    assert len(log) == 2
    assert all(entry["reasoning"] for entry in log)


def test_router_never_writes_to_semantic_memory():
    """Hard requirement from the lab spec: this router only ever produces
    forget/promote-to-episodic decisions. It has no semantic-memory
    write path at all."""
    router = PromoteDropRouter()
    assert not hasattr(router, "semantic_store")
    assert not hasattr(router, "write_semantic")
