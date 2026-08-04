class ZonePruner:
    def __init__(self, head_size=2, tail_size=10, middle_keep_ratio=0.2):
        self.head_size = head_size
        self.tail_size = tail_size
        self.middle_keep_ratio = middle_keep_ratio

    def apply(self, messages):
        n = len(messages)

        if n <= self.head_size + self.tail_size:
            return messages

        head = messages[:self.head_size]
        tail = messages[-self.tail_size:]
        middle = messages[self.head_size:-self.tail_size]

        middle_keep_count = max(
            1,
            int(len(middle) * self.middle_keep_ratio)
        ) if middle else 0

        kept_middle = self._sample_middle(middle, middle_keep_count)

        return head + kept_middle + tail

    def _sample_middle(self, middle, keep_count):
        if keep_count >= len(middle):
            return middle

        step = len(middle) / keep_count
        indices = [int(i * step) for i in range(keep_count)]

        return [middle[i] for i in indices]


def zone_prune(messages, head_size=2, tail_size=10, middle_keep_ratio=0.2):
    pruner = ZonePruner(
        head_size=head_size,
        tail_size=tail_size,
        middle_keep_ratio=middle_keep_ratio
    )
    return pruner.apply(messages)