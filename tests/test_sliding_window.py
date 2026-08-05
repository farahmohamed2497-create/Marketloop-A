import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from mcp_server.memory.sliding_window import apply_sliding_window


def test_keeps_last_n_messages():
    messages = [{"role": "user", "content": str(i)} for i in range(20)]

    result = apply_sliding_window(messages, window_size=5)

    assert len(result) == 5
    assert result[0]["content"] == "15"
    assert result[-1]["content"] == "19"


def test_no_op_when_under_window_size():
    messages = [{"role": "user", "content": str(i)} for i in range(3)]

    result = apply_sliding_window(messages, window_size=10)

    assert result == messages


def test_drops_early_critical_detail_on_long_transcript():
    """Documents the known weakness: an early return reason is lost once
    the transcript exceeds the window - this is exactly what shows up in
    the comparison table's accuracy column."""
    messages = [{"role": "user", "content": "reason: damaged in shipping"}]
    messages += [{"role": "tool", "content": f"noise_{i}"} for i in range(40)]

    result = apply_sliding_window(messages, window_size=10)

    assert "damaged in shipping" not in " ".join(m["content"] for m in result)
