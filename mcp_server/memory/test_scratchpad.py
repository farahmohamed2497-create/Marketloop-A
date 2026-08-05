import pytest

from mcp_server.memory.scratchpad import Scratchpad
from mcp_server.memory.masking import mask_tool_outputs
from mcp_server.memory.rolling_buffer import RollingBuffer
from mcp_server.memory.summarization import RecursiveSummarizer


# ---------- Scratchpad -------

def test_set_plan_updates_plan_and_timestamp():
    pad = Scratchpad()
    pad.set_plan("Check appointment history then prescriptions")

    assert pad.plan == "Check appointment history then prescriptions"
    assert pad.updated_at is not None


def test_set_sub_goal_updates_sub_goal():
    pad = Scratchpad()
    pad.set_sub_goal("Fetch prescriptions for pet_id=42")

    assert pad.sub_goal == "Fetch prescriptions for pet_id=42"


def test_update_state_and_get_state():
    pad = Scratchpad()
    pad.update_state("pet_id", 42)

    assert pad.get_state("pet_id") == 42


def test_get_state_returns_default_when_missing():
    pad = Scratchpad()

    assert pad.get_state("missing_key", default="none") == "none"


def test_snapshot_returns_full_state():
    pad = Scratchpad()
    pad.set_plan("plan A")
    pad.set_sub_goal("sub-goal A")
    pad.update_state("step", 1)

    snap = pad.snapshot()

    assert snap["plan"] == "plan A"
    assert snap["sub_goal"] == "sub-goal A"
    assert snap["state"] == {"step": 1}
    assert snap["updated_at"] is not None


def test_clear_resets_everything():
    pad = Scratchpad()
    pad.set_plan("plan A")
    pad.set_sub_goal("sub-goal A")
    pad.update_state("step", 1)

    pad.clear()

    assert pad.plan is None
    assert pad.sub_goal is None
    assert pad.state == {}


# ---------- (Pruning Isolation)### ----------

def _build_transcript():
    messages = [{
        "role": "user",
        "content": "Customer allergy is peanuts"
    }]
    for i in range(50):
        messages.append({
            "role": "tool",
            "content": f"tool output {i}"
        })
    return messages


def test_pruning_masking_does_not_touch_scratchpad():
    pad = Scratchpad()
    pad.set_plan("Diagnose allergy risk")
    pad.set_sub_goal("Check prescription conflicts")
    pad.update_state("pet_id", 42)

    before = pad.snapshot()
    messages = _build_transcript()

    mask_tool_outputs(messages, keep_last_outputs=3)

    after = pad.snapshot()
    assert before == after


def test_pruning_rolling_buffer_does_not_touch_scratchpad():
    pad = Scratchpad()
    pad.set_plan("Diagnose allergy risk")
    pad.set_sub_goal("Check prescription conflicts")

    before = pad.snapshot()
    messages = _build_transcript()

    buffer = RollingBuffer(max_turns=10)
    for msg in messages:
        buffer.add_turn(msg["role"], msg["content"])

    after = pad.snapshot()
    assert before == after


def test_pruning_summarization_does_not_touch_scratchpad():
    pad = Scratchpad()
    pad.set_plan("Diagnose allergy risk")
    pad.set_sub_goal("Check prescription conflicts")

    before = pad.snapshot()
    messages = _build_transcript()

    summarizer = RecursiveSummarizer(threshold=10)
    summarizer.summarize(messages)

    after = pad.snapshot()
    assert before == after


def test_scratchpad_survives_even_when_buffer_is_fully_pruned():
    
    pad = Scratchpad()
    pad.set_sub_goal("Waiting on prescription lookup")

    buffer = RollingBuffer(max_turns=1)
    for msg in _build_transcript():
        buffer.add_turn(msg["role"], msg["content"])

    
    assert len(buffer.get_context()) == 1


    assert pad.sub_goal == "Waiting on prescription lookup"