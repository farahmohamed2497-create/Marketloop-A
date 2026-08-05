import pytest

from mcp_server.memory.zone_pruning import ZonePruner, zone_prune
from mcp_server.memory.scratchpad import Scratchpad


def _build_transcript(n_middle=50):
    messages = [{
        "role": "user",
        "content": "Customer allergy is peanuts"
    }]
    for i in range(n_middle):
        messages.append({
            "role": "tool",
            "content": f"tool output {i}"
        })
    for i in range(10):
        messages.append({
            "role": "assistant",
            "content": f"recent message {i}"
        })
    return messages




def test_returns_all_messages_if_below_threshold():
    messages = [{"role": "user", "content": "hi"}] * 5

    result = zone_prune(messages, head_size=2, tail_size=10)

    assert result == messages


def test_head_is_always_preserved():
    messages = _build_transcript()

    result = zone_prune(messages, head_size=2, tail_size=10, middle_keep_ratio=0.2)

    assert result[0] == messages[0]
    assert result[1] == messages[1]


def test_tail_is_always_preserved():
    messages = _build_transcript()
    tail_size = 10

    result = zone_prune(messages, head_size=2, tail_size=tail_size, middle_keep_ratio=0.2)

    assert result[-tail_size:] == messages[-tail_size:]


def test_middle_is_sampled_not_fully_kept():
    messages = _build_transcript(n_middle=50)

    result = zone_prune(messages, head_size=2, tail_size=10, middle_keep_ratio=0.2)

  
    assert len(result) < len(messages)
    assert len(result) > 2 + 10  


def test_middle_keep_ratio_zero_keeps_only_head_and_tail():
    messages = _build_transcript(n_middle=50)

    result = zone_prune(messages, head_size=2, tail_size=10, middle_keep_ratio=0.0)

    assert len(result) <= 2 + 1 + 10


def test_critical_detail_in_head_always_survives():
    messages = _build_transcript(n_middle=100)

    result = zone_prune(messages, head_size=2, tail_size=10, middle_keep_ratio=0.1)

    assert "peanuts" in str(result)


def test_small_transcript_shorter_than_head_plus_tail_returns_unchanged():
    messages = _build_transcript(n_middle=1)  

    result = zone_prune(messages, head_size=2, tail_size=10)

    assert result == messages


# ---------- (Pruning Isolation) ----------

def test_zone_pruning_does_not_touch_scratchpad():
    pad = Scratchpad()
    pad.set_plan("Diagnose allergy risk")
    pad.set_sub_goal("Check prescription conflicts")
    pad.update_state("pet_id", 42)

    before = pad.snapshot()
    messages = _build_transcript()

    zone_prune(messages, head_size=2, tail_size=10, middle_keep_ratio=0.2)

    after = pad.snapshot()
    assert before == after


def test_zone_pruner_does_not_mutate_original_messages():
    messages = _build_transcript()
    original_length = len(messages)

    zone_prune(messages, head_size=2, tail_size=10, middle_keep_ratio=0.2)

    #(immutability)
    assert len(messages) == original_length