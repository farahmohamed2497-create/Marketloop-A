class RecursiveSummarizer:
    def __init__(self, threshold=10):
        self.threshold = threshold

    def summarize(self, messages):
        if len(messages) <= self.threshold:
            return messages

        older = messages[:-self.threshold]
        recent = messages[-self.threshold:]

        important = []

        for msg in older:
            if msg["role"] == "user":
                important.append(msg["content"])

        summary_text = " | ".join(
            important[:8]
        )

        return [
            {
                "role": "system",
                "content": summary_text
            }
        ] + recent