"""Naive RAG retrieval baseline for MarketLoop.

This is intentionally a single vector-search round: no BM25 fusion, query
decomposition, or retry loop.  It is the baseline used by retrieval_eval to
measure the value of Hybrid and Agentic RAG.
"""

from __future__ import annotations

from typing import Any

from RAG.embedding import EmbeddingModel
from RAG.vector_store import VectorStore


class NaiveRAGRetriever:
    """Embed one query and return its nearest vector-store chunks."""

    def __init__(self, embedder: EmbeddingModel, vector_store: VectorStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run the baseline chunk -> embed -> ANN retrieve path."""
        query_embedding = self.embedder.embed(query)
        return self.vector_store.search(
            query_embedding=query_embedding,
            k=top_k,
            filters=filters,
        )