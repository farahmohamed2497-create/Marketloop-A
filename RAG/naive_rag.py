"""Naive RAG retrieval baseline for MarketLoop."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from RAG.embedding import EmbeddingModel

if TYPE_CHECKING:
    from RAG.vector_store import VectorStore


class NaiveRAGRetriever:
    """Embed one query and return its nearest vector-store chunks."""

    def __init__(
        self,
        embedder: EmbeddingModel,
        vector_store: "VectorStore",
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        query_embedding = self.embedder.embed(query)

        return self.vector_store.search(
            query_embedding=query_embedding,
            k=top_k,
            filters=filters,
        )