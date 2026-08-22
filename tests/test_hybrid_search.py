from RAG.hybrid_search import HybridSearch


def test_hybrid_returns_results():

    class FakeVectorStore:
        @staticmethod
        def search(emb, k):
            return [
                {
                    "text": "gaming laptop",
                    "score": 0.9,
                    "metadata": {}
                }
            ]

    class FakeKeywordStore:
        @staticmethod
        def query(text, top_k):
            return [
                {
                    "payload": "gaming laptop",
                    "score": 1.0,
                    "metadata": {}
                }
            ]

    hybrid = HybridSearch(
        FakeVectorStore(),
        FakeKeywordStore()
    )

    results = hybrid.search(
        [0.1, 0.2],
        "gaming laptop"
    )

    assert len(results) > 0
