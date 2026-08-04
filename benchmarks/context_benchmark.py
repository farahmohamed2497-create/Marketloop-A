import time

from benchmarks.token_counter import count_tokens
from mcp_server.memory.masking import mask_tool_outputs
from mcp_server.memory.rolling_buffer import RollingBuffer
from mcp_server.memory.summarization import RecursiveSummarizer

messages = [{
    "role": "user",
    "content": "Customer allergy is peanuts"
}]

for i in range(100):
    messages.append({
        "role": "tool",
        "content": f"tool output {i}"
    })


# Sliding Window
start = time.perf_counter()

buffer = RollingBuffer(max_turns=10)

for msg in messages:
    buffer.add_turn(
        msg["role"],
        msg["content"]
    )

elapsed = time.perf_counter() - start

print(
    "Sliding Window",
    len(str(buffer.get_context()).split()),
    elapsed
)

# Masking
start = time.perf_counter()

masked = mask_tool_outputs(messages)

elapsed = time.perf_counter() - start

print(
    "Observation Masking",
    len(str(masked).split()),
    elapsed
)

# Summarization
start = time.perf_counter()

summarizer = RecursiveSummarizer()

summary = summarizer.summarize(messages)

elapsed = time.perf_counter() - start

print(
    "Recursive Summarization",
    len(str(summary).split()),
    elapsed
)

print(
    "Sliding Window Tokens:",
    count_tokens(buffer.get_context())
)

print(
    "Observation Masking Tokens:",
    count_tokens(masked)
)

print(
    "Recursive Summarization Tokens:",
    count_tokens(summary)
)
