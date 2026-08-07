class HybridSearch:

    def __init__(self, vector_store, keyword_store):
        self.vector_store = vector_store
        self.keyword_store = keyword_store

    def search(
        self,
        query_embedding,
        query_text,
        top_k=5,
        alpha=0.7
    ):

        vector_results = self.vector_store.search(
            query_embedding,
            k=top_k * 2
        )

        keyword_results = self.keyword_store.query(
            query_text,
            top_k=top_k * 2
        )

        scores = {}

        for r in vector_results:
            text = r["text"]

            scores[text] = {
                "text": text,
                "metadata": r["metadata"],
                "score": alpha * r["score"]
            }

        for r in keyword_results:
            text = r["payload"]

            if text not in scores:
                scores[text] = {
                    "text": text,
                    "metadata": r["metadata"],
                    "score": 0
                }

            scores[text]["score"] += (1 - alpha) * r["score"]

        ranked = sorted(
            scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        return ranked[:top_k]