class RecursiveSummarizer:
    """
    Context strategy #3: compress older turns into a running summary every
    `threshold` turns, keeping the most recent turns verbatim.

    Only user-stated facts are folded into the summary (tool output noise
    is dropped entirely) because in a MarketLoop support call, the details
    that matter for a final decision - e.g. "item arrived damaged" -
    always come from the customer, never from a tool result.
    """

    def __init__(self, threshold=10):
        self.threshold = threshold

    def summarize(self, messages):
        if len(messages) <= self.threshold:
            return messages

        older = messages[:-self.threshold]
        recent = messages[-self.threshold:]

        important = [
            msg["content"] for msg in older
            if msg.get("role") == "user"
        ]

        # NOTE: the original version capped this at important[:8], which
        # silently drops any user fact stated after the 8th one - in a
        # long call that's exactly where a late but critical detail (like
        # a return reason mentioned mid-conversation) would get lost.
        # Compression here comes from stripping tool-output noise, not
        # from truncating the facts we already decided were worth keeping.
        summary_text = " | ".join(important)

        return [
            {
                "role": "system",
                "content": f"[Summary of {len(older)} earlier turns] {summary_text}"
            }
        ] + recent