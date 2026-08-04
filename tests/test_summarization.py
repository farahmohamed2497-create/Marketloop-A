from mcp_server.memory.summarization import RecursiveSummarizer


def test_old_messages_are_compressed():
    messages = []

    for i in range(20):
        messages.append({
            "role": "user",
            "content": f"message_{i}"
        })

    summarizer = RecursiveSummarizer(threshold=10)

    result = summarizer.summarize(messages)

    assert result[0]["role"] == "system"
    assert "Summary" in result[0]["content"]
    assert len(result) == 11