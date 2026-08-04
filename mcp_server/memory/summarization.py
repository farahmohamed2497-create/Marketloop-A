class RecursiveSummarizer:
    def __init__(self, threshold=10):
        self.threshold = threshold

    def summarize(self, messages):
        if len(messages) <= self.threshold:
            return messages

        older = messages[:-self.threshold]
        recent = messages[-self.threshold:]

        summary_text = (
            f"Summary of {len(older)} previous messages"
        )

        return [
            {
                "role": "system",
                "content": summary_text
            }
        ] + recent