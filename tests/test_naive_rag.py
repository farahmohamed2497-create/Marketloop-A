from rag.naive_rag import NaiveRAGRetriever


class FakeEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed(self, query: str) -> list[float]:
        self.queries.append(query)
        return [0.1, 0.2]


class FakeVectorStore:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search(self, *, query_embedding, k, filters=None):
        self.calls.append(
            {"query_embedding": query_embedding, "k": k, "filters": filters}
        )
        return [{"text": "UltraView warranty is two years.", "score": 0.9, "metadata": {}}]


def test_naive_rag_embeds_once_and_runs_one_vector_search():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()
    retriever = NaiveRAGRetriever(embedder, vector_store)

    results = retriever.retrieve(
        "What warranty applies to SKU ELEC-001?",
        top_k=2,
        filters={"doc": "catalog"},
    )

    assert embedder.queries == ["What warranty applies to SKU ELEC-001?"]
    assert vector_store.calls == [
        {"query_embedding": [0.1, 0.2], "k": 2, "filters": {"doc": "catalog"}}
    ]
    assert results[0]["text"] == "UltraView warranty is two years."
