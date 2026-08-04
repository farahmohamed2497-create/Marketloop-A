from mcp_server.memory.rolling_buffer import RollingBuffer


def test_buffer_keeps_last_n_turns():
    buffer = RollingBuffer(max_turns=3)

    buffer.add_turn("user", "1")
    buffer.add_turn("user", "2")
    buffer.add_turn("user", "3")
    buffer.add_turn("user", "4")

    context = buffer.get_context()

    assert len(context) == 3
    assert context[0]["content"] == "2"
    assert context[-1]["content"] == "4"