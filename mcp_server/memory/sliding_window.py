"""
Context strategy #1: Sliding window.

This is deliberately kept separate from rolling_buffer.py even though the
core mechanism (keep last N turns) is similar:

- rolling_buffer.py is short-term memory: it is *always on*, storing the
  live conversation as it happens (Concern: "Short-term memory and
  scratchpad").
- sliding_window.py is one candidate *pruning strategy* evaluated in
  context_eval/ against three others (masking, summarization, and
  zone-based pruning) to decide what MarketLoop ships when a transcript
  needs to be cut down for an LLM call (Concern: "Context window
  management, all four strategies").

Keeping them as separate modules means the comparison harness can swap
strategies in and out without touching the always-on short-term buffer.
"""

from __future__ import annotations

from typing import Any


def apply_sliding_window(
    messages: list[dict[str, Any]],
    window_size: int = 10,
) -> list[dict[str, Any]]:
    """Keep only the last `window_size` messages, dialogue and tool
    output alike. Cheapest strategy, but blind to *which* messages
    matter - a customer's return reason stated early in a long call is
    dropped just as readily as stale tool output."""
    if len(messages) <= window_size:
        return messages
    return messages[-window_size:]
