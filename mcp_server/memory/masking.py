class ObservationMasker:
    """
    Context strategy #2: keep dialogue turns intact, drop older tool-call
    outputs (order lookups, shipment checks, inventory checks...) since a
    MarketLoop support call can rack up 30+ tool calls while the customer's
    original detail (e.g. return reason) sits in the *dialogue*, not in the
    tool outputs. Masking targets the actual bloat source instead of
    trimming turns uniformly like sliding window does.
    """

    def __init__(self, keep_last_outputs: int = 3):
        self.keep_last_outputs = keep_last_outputs

    def apply(self, messages):
        # index-based tracking instead of id(msg): id() keys on object
        # identity, which breaks if a message dict is copied/rebuilt
        # (e.g. after JSON round-tripping through an MCP tool call) - two
        # logically-identical messages would then get different ids and
        # the wrong ones could be kept/dropped. Position in the transcript
        # is the actual thing we care about.
        tool_indices = [
            i for i, msg in enumerate(messages)
            if msg.get("role") == "tool"
        ]
        keep_indices = set(tool_indices[-self.keep_last_outputs:])

        result = []
        for i, msg in enumerate(messages):
            if msg.get("role") != "tool":
                result.append(msg)
            elif i in keep_indices:
                result.append(msg)

        return result


def mask_tool_outputs(messages, keep_last_outputs=3):
    masker = ObservationMasker(
        keep_last_outputs=keep_last_outputs
    )
    return masker.apply(messages)