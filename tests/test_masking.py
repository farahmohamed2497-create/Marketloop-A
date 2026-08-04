from mcp_server.memory.masking import ObservationMasker


def test_keep_last_three_tool_outputs():
    messages = []

    for i in range(5):
        messages.append({
            "role": "tool",
            "content": f"output_{i}"
        })

    masker = ObservationMasker(keep_last_outputs=3)

    result = masker.apply(messages)

    assert len(result) == 3
    assert result[0]["content"] == "output_2"
    assert result[-1]["content"] == "output_4"