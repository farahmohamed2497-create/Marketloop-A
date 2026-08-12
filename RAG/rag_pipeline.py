"""RAG response paths for MarketLoop.

External services are initialized lazily so this module can be imported by
tests without a Groq API key or pre-built local retrieval indexes.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from rag.agentic_rag import AgenticRAGRetriever
from rag.embedding import EmbeddingModel
from rag.hybrid_search import HybridSearch
from rag.naive_rag import NaiveRAGRetriever
from rag.self_rag_verification import check_relevance, check_support
from rag.vector_store import VectorStore


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Kept as module attributes so integration tests can monkeypatch them with
# deterministic fakes.  Do not create clients or load indexes at import time.
_groq_client: Groq | None = None
_embedder: EmbeddingModel | None = None
_naive: NaiveRAGRetriever | None = None
_hybrid: HybridSearch | None = None
_agentic: AgenticRAGRetriever | None = None


class _RenamingUnpickler(pickle.Unpickler):
    """Load keyword-store pickles created before package-name normalization."""

    def find_class(self, module: str, name: str) -> Any:
        # The repository package is named RAG.  Accept old lowercase pickles.
        if module == "rag" or module.startswith("rag."):
            module = "RAG" + module[len("rag"):]
        return super().find_class(module, name)


def _load_stores() -> tuple[EmbeddingModel, NaiveRAGRetriever, HybridSearch, AgenticRAGRetriever]:
    """Load local indexes only when a real, non-test request needs them."""
    global _embedder, _naive, _hybrid, _agentic

    if _embedder is not None and _naive is not None and _hybrid is not None and _agentic is not None:
        return _embedder, _naive, _hybrid, _agentic

    embedder = EmbeddingModel()
    vector_store = VectorStore.load(str(PROJECT_ROOT / "data" / "marketloop_vector_db"))
    with (PROJECT_ROOT / "data" / "keyword_store.pkl").open("rb") as handle:
        keyword_store = _RenamingUnpickler(handle).load()

    _embedder = embedder
    _naive = NaiveRAGRetriever(embedder=embedder, vector_store=vector_store)
    _hybrid = HybridSearch(vector_store=vector_store, keyword_store=keyword_store)
    def hybrid_search_for_agentic(query: str, top_k: int):
        return _hybrid.search(
            query_embedding=_embedder.embed(query),
            query_text=query,
            top_k=top_k,
        )

    _agentic = AgenticRAGRetriever(search_fn=hybrid_search_for_agentic)
    return _embedder, _naive, _hybrid, _agentic


def _get_groq_client() -> Groq:
    """Create the Groq client only for a real generation request."""
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not configured. Add it to .env before running "
                "a real RAG generation request. Tests use a fake client and do not need it."
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def _generate(query: str, context: str) -> str:
    prompt = f"""You are an assistant.

Answer ONLY using the provided context.

Context:
{context}

Question:
{query}

Answer:
"""
    response = _get_groq_client().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


def answer_with_naive(query: str, top_k: int = 3) -> dict[str, Any]:
    """Single-round vector baseline plus the same verification steps as Hybrid."""
    global _naive
    if _naive is None:
        _load_stores()
    assert _naive is not None

    results = _naive.retrieve(query, top_k=top_k)
    if not results:
        return {
            "answer": "No relevant information found in the knowledge base.",
            "chunks": [],
            "relevance_checks": [],
            "support_check": None,
        }

    relevance_checks = [check_relevance(query, item["text"]) for item in results]
    relevant_results = [item for item, check in zip(results, relevance_checks) if check.passed]
    if not relevant_results:
        return {
            "answer": "Retrieved content did not pass relevance verification.",
            "chunks": results,
            "relevance_checks": relevance_checks,
            "support_check": None,
        }

    context = "\n\n".join(item["text"] for item in relevant_results)
    answer = _generate(query, context)
    support_check = check_support(answer, [item["text"] for item in relevant_results])

    if not support_check.passed:
        answer = (
            "I can't provide a grounded answer because the generated response "
            "was not sufficiently supported by the retrieved company documents."
        )

    return {
        "answer": answer,
        "chunks": relevant_results,
        "relevance_checks": relevance_checks,
        "support_check": support_check,
    }


def answer_with_hybrid(query: str, top_k: int = 3) -> dict[str, Any]:
    """Hybrid retrieval (vector + BM25), generation, and Self-RAG checks."""
    global _embedder, _hybrid
    if _embedder is None or _hybrid is None:
        _load_stores()
    assert _embedder is not None and _hybrid is not None

    results = _hybrid.search(
        query_embedding=_embedder.embed(query),
        query_text=query,
        top_k=top_k,
    )
    if not results:
        return {
            "answer": "No relevant information found in the knowledge base.",
            "chunks": [],
            "relevance_checks": [],
            "support_check": None,
        }

    relevance_checks = [check_relevance(query, item["text"]) for item in results]
    relevant_results = [item for item, check in zip(results, relevance_checks) if check.passed]
    if not relevant_results:
        return {
            "answer": "Retrieved content did not pass relevance verification.",
            "chunks": results,
            "relevance_checks": relevance_checks,
            "support_check": None,
        }

    context = "\n\n".join(item["text"] for item in relevant_results)
    answer = _generate(query, context)
    support_check = check_support(answer, [item["text"] for item in relevant_results])

    if not support_check.passed:
        answer = (
            "I can't provide a grounded answer because the generated response "
            "was not sufficiently supported by the retrieved company documents."
        )
    return {
        "answer": answer,
        "chunks": relevant_results,
        "relevance_checks": relevance_checks,
        "support_check": support_check,
    }


def answer_with_agentic(query: str) -> dict[str, Any]:
    """Agentic retrieval, generation, and Self-RAG support verification."""
    global _agentic
    if _agentic is None:
        _load_stores()
    assert _agentic is not None

    result = _agentic.run(query)
    if not result.final_chunks:
        return {
            "answer": "No relevant information found for any sub-question.",
            "hops": result.hops,
            "support_check": None,
        }

    chunk_texts = [chunk["payload"] for chunk in result.final_chunks]
    answer = _generate(query, "\n\n".join(chunk_texts))
    support_check = check_support(answer, chunk_texts)
    if not support_check.passed:
        answer = (
            "I can't provide a grounded answer because the generated response "
            "was not sufficiently supported by the retrieved company documents."
        )
    return {"answer": answer, "hops": result.hops, "support_check": support_check}