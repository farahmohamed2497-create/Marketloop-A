from collections import deque


class RollingBuffer:
    """
    Short-term memory: the live, always-on transcript of the current
    MarketLoop support call (customer messages + tool outputs), separate
    from the Scratchpad (scratchpad.py) which tracks the agent's current
    goal/working state.

    This is the raw material that context-management strategies
    (sliding_window.py, masking.py, summarization.py) operate on when the
    transcript needs to be cut down for an LLM call - the buffer itself
    just accumulates turns; it doesn't decide what to prune.
    """

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.buffer = deque(maxlen=max_turns)

    def add_turn(self, role: str, content: str):
        self.buffer.append({
            "role": role,
            "content": content
        })

    def get_context(self):
        return list(self.buffer)

    def clear(self):
        self.buffer.clear()