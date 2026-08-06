from collections import defaultdict


class MetadataIndex:
    def __init__(self):
        self.index = defaultdict(set)

    def add(self, doc_id, metadata):
        for key, value in metadata.items():
            self.index[(key, value)].add(doc_id)

    def filter_ids(self, filters):
        if not filters:
            return None

        results = []

        for key, value in filters.items():
            results.append(
                self.index.get((key, value), set())
            )

        if not results:
            return set()

        return set.intersection(*results)