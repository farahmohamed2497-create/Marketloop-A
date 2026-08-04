from mcp_server.memory.rolling_buffer import RollingBuffer
from mcp_server.memory.masking import mask_tool_outputs
from mcp_server.memory.summarization import RecursiveSummarizer


def build_transcript():
    messages = [{
        "role": "user",
        "content": "Customer allergy is peanuts"
    }]

    for i in range(50):
        messages.append({
            "role": "tool",
            "content": f"tool output {i}"
        })

    messages.append({
        "role": "user",
        "content": "What is my allergy?"
    })

    return messages


def test_long_context_buffer():
    buffer = RollingBuffer(max_turns=10)

    for msg in build_transcript():
        buffer.add_turn(msg["role"], msg["content"])

    assert len(buffer.get_context()) <= 10


def test_long_context_masking():
    messages = build_transcript()

    result = mask_tool_outputs(messages)

    tool_count = len(
        [m for m in result if m["role"] == "tool"]
    )

    assert tool_count <= 3


def test_long_context_summary():
    summarizer = RecursiveSummarizer()

    result = summarizer.summarize(
        build_transcript()
    )

    assert len(result) < len(build_transcript())