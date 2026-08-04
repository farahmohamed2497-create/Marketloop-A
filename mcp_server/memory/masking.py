class ObservationMasker:
    def __init__(self, keep_last_outputs: int = 3):
        self.keep_last_outputs = keep_last_outputs

    def apply(self, messages):
        tool_messages = [
            msg for msg in messages
            if msg.get("role") == "tool"
        ]

        allowed_ids = set(
            id(msg)
            for msg in tool_messages[-self.keep_last_outputs:]
        )

        result = []

        for msg in messages:
            if msg.get("role") != "tool":
                result.append(msg)
                continue

            if id(msg) in allowed_ids:
                result.append(msg)

        return result


def mask_tool_outputs(messages, keep_last_outputs=3):
    masker = ObservationMasker(
        keep_last_outputs=keep_last_outputs
    )
    return masker.apply(messages)