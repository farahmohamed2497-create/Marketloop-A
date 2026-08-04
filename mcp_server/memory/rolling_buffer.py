from collections import deque


class RollingBuffer:
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