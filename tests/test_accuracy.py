import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from typing import Any

from mcp_server.memory.rolling_buffer import RollingBuffer
from mcp_server.memory.masking import ObservationMasker
from mcp_server.memory.summarization import RecursiveSummarizer


FACTS = [
    "Customer allergy is peanuts",
    "Customer city is Alexandria",
    "Customer favorite color is blue",
    "Customer order number is 12345",
    "Customer phone is 01012345678",
    "Customer email is test@example.com",
    "Customer membership is gold",
    "Customer preferred payment is visa",
    "Customer last purchase was laptop",
    "Customer birthday is January"
]


def build_accuracy_transcript():
    messages = []

    for fact in FACTS:
        messages.append({
            "role": "user",
            "content": fact
        })

        for i in range(20):
            messages.append({
                "role": "tool",
                "content": f"tool output {i}"
            })

    return messages


def recall_score(messages):
    text = " ".join(
        msg["content"]
        for msg in messages
    )

    score = 0

    for fact in FACTS:
        keyword = fact.split()[-1]

        if keyword in text:
            score += 1

    return score


def test_rolling_buffer_accuracy():
    buffer = RollingBuffer(max_turns=120)

    for msg in build_accuracy_transcript():
        buffer.add_turn(
            msg["role"],
            msg["content"]
        )

    result = buffer.get_context()

    score = recall_score(result)

    print(f"Rolling Buffer Accuracy: {score}/10")

    assert 0 <= score <= 10


def test_masking_accuracy():
    messages = build_accuracy_transcript()

    masker = ObservationMasker()

    result: list[Any] = masker.apply(messages)

    score = recall_score(result)

    print(f"Observation Masking Accuracy: {score}/10")

    assert 0 <= score <= 10


def test_summarization_accuracy():
    summarizer = RecursiveSummarizer()

    result = summarizer.summarize(
        build_accuracy_transcript()
    )

    score = recall_score(result)

    print(f"Recursive Summarization Accuracy: {score}/10")

    assert 0 <= score <= 10