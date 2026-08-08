"""Compare Naive, Hybrid, and Agentic RAG on one fixed MarketLoop test set.

Metrics are retrieval-only and reproducible: accuracy is evidence coverage in
retrieved chunks, tokens are the word count of verification-approved chunks,
and latency is measured per query after the common indexes are built.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Callable

sys.path.append(str(Path(__file__).resolve().parents[1]))

from RAG.agentic_rag import AgenticRAGRetriever
from RAG.chunking import chunk_document
from RAG.embedding import EmbeddingModel
from RAG.hybrid_search import HybridSearch
from RAG.naive_rag import NaiveRAGRetriever
from RAG.self_rag_verification import check_relevance
from RAG.vector_store import VectorStore
from mcp_server.tools.knowledge_store import KeywordStore
from retrieval_eval.evaluation_questions import (
    RETRIEVAL_QUESTIONS,
    expected_answer_is_retrieved,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "Complete Enterprise Product Catalog.txt"
Retriever = Callable[[str], list[dict[str, Any]]]


def _chunk_text(result: dict[str, Any]) -> str:
    return result.get("text", result.get("payload", ""))


def _count_tokens(chunks: list[str]) -> int:
    return sum(len(chunk.split()) for chunk in chunks)


def build_retrievers() -> tuple[Retriever, Retriever, Retriever]:
    """Build identical vector and BM25 indexes once for all architectures."""
    catalog = CATALOG_PATH.read_text(encoding="utf-8")
    chunks = chunk_document(
        catalog,
        base_metadata={"doc": "Complete Enterprise Product Catalog"},
    )

    embedder = EmbeddingModel()
    vector_store = VectorStore(dim=embedder.dim, max_elements=len(chunks) + 10)
    keyword_store = KeywordStore()

    embeddings = embedder.embed_batch([chunk["text"] for chunk in chunks])
    for chunk, embedding in zip(chunks, embeddings):
        vector_store.add(
            text=chunk["text"],
            embedding=embedding,
            metadata=chunk["metadata"],
        )
        keyword_store.upsert(payload=chunk["text"], metadata=chunk["metadata"])

    naive = NaiveRAGRetriever(embedder, vector_store)
    hybrid = HybridSearch(vector_store, keyword_store)

    def naive_retrieve(query: str) -> list[dict[str, Any]]:
        return naive.retrieve(query, top_k=3)

    def hybrid_retrieve(query: str) -> list[dict[str, Any]]:
        return hybrid.search(
            query_embedding=embedder.embed(query),
            query_text=query,
            top_k=3,
        )

    # Agentic RAG plans and observes each hop; Hybrid is its underlying
    # retriever so exact identifiers and semantic similarity both work.
    def hybrid_search_fn(query: str, top_k: int) -> list[dict[str, Any]]:
        return hybrid.search(
            query_embedding=embedder.embed(query),
            query_text=query,
            top_k=top_k,
        )

    agentic = AgenticRAGRetriever(search_fn=hybrid_search_fn, top_k=3)

    def agentic_retrieve(query: str) -> list[dict[str, Any]]:
        return agentic.run(query).final_chunks

    return naive_retrieve, hybrid_retrieve, agentic_retrieve


def evaluate_architecture(name: str, retrieve: Retriever) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    correct = 0
    token_total = 0
    latency_total_ms = 0.0
    details: list[dict[str, Any]] = []

    for question in RETRIEVAL_QUESTIONS:
        started = time.perf_counter()
        raw_results = retrieve(question["query"])
        latency_ms = (time.perf_counter() - started) * 1000

        approved_chunks = [
            text
            for item in raw_results
            if (text := _chunk_text(item))
            and check_relevance(question["query"], text).passed
        ]
        hit = expected_answer_is_retrieved(question, approved_chunks)
        correct += int(hit)
        token_total += _count_tokens(approved_chunks)
        latency_total_ms += latency_ms
        details.append(
            {
                "id": question["id"],
                "category": question["category"],
                "passed": hit,
                "approved_chunks": len(approved_chunks),
            }
        )

    total = len(RETRIEVAL_QUESTIONS)
    return (
        {
            "architecture": name,
            "accuracy": f"{correct}/{total}",
            "avg_tokens": round(token_total / total, 1),
            "avg_latency_ms": round(latency_total_ms / total, 3),
        },
        details,
    )


def main() -> list[dict[str, Any]]:
    naive, hybrid, agentic = build_retrievers()
    evaluations = [
        ("Naive RAG (vector only)", naive),
        ("Hybrid Search (vector + BM25)", hybrid),
        ("Agentic RAG (Hybrid multi-hop)", agentic),
    ]

    rows: list[dict[str, Any]] = []
    print(f"\nFixed retrieval test set: {len(RETRIEVAL_QUESTIONS)} questions\n")
    for name, retrieve in evaluations:
        row, details = evaluate_architecture(name, retrieve)
        rows.append(row)
        passed = sum(item["passed"] for item in details)
        print(f"{name}: {passed}/{len(details)} evidence checks passed")

    print("\n| Architecture | Accuracy | Avg tokens/query | Avg latency (ms) |")
    print("|---|---|---:|---:|")
    for row in rows:
        print(
            f"| {row['architecture']} | {row['accuracy']} | "
            f"{row['avg_tokens']} | {row['avg_latency_ms']} |"
        )
    return rows


if __name__ == "__main__":
    main()