"""
rag/rag_pipeline.py

Wires hybrid search + agentic RAG + generation + Self-RAG verification
into callable functions, ready to be used from agent/client.py.
"""

from __future__ import annotations

import os
import pickle
from dotenv import load_dotenv
from groq import Groq

from rag.embedding import EmbeddingModel
from rag.vector_store import VectorStore
from rag.hybrid_search import HybridSearch
from rag.agentic_rag import AgenticRAGRetriever
from rag.self_rag_verification import check_relevance, check_support
from mcp_server.tools.knowledge_store import KeywordStore

load_dotenv()

_groq_client = Groq(api_key=os.getenv("GROQ_API_KEY1"))


class _RenamingUnpickler(pickle.Unpickler):
    """
    Unpickler متوافق مع ملفات .pkl القديمة اللي اتعملها save وقت
    ما كان اسم الفولدر 'RAG' (بحرف كبير)، قبل ما يتعمله rename لـ 'rag'.
    بيحوّل أي اسم موديول بيبدأ بـ RAG لنفس الاسم بحرف صغير تلقائيًا.
    """

    def find_class(self, module, name):
        if module == "RAG" or module.startswith("RAG."):
            module = "rag" + module[len("RAG"):]
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            # fallback: لو الكلاس أصلاً اتنقل دلوقتي لمكان تاني (مش جوه rag)
            if name == "KeywordStore":
                from mcp_server.tools.knowledge_store import KeywordStore
                return KeywordStore
            raise


def _load_stores():
    embedder = EmbeddingModel()
    vector_store = VectorStore.load("./data/marketloop_vector_db")

    with open("./data/keyword_store.pkl", "rb") as f:
        keyword_store = _RenamingUnpickler(f).load()

    hybrid = HybridSearch(vector_store=vector_store, keyword_store=keyword_store)
    agentic = AgenticRAGRetriever(store=keyword_store)
    return embedder, hybrid, agentic


_embedder, _hybrid, _agentic = _load_stores()


def _generate(query: str, context: str) -> str:
    prompt = f"""
You are an assistant.

Answer ONLY using the provided context.

Context:
{context}

Question:
{query}

Answer:
"""
    response = _groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


def answer_with_hybrid(query: str, top_k: int = 3) -> dict:
    """Hybrid retrieval (vector + BM25) + generation + Self-RAG checks."""
    query_embedding = _embedder.embed(query)

    results = _hybrid.search(
        query_embedding=query_embedding,
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

    relevance_checks = [check_relevance(query, r["text"]) for r in results]
    relevant_results = [r for r, c in zip(results, relevance_checks) if c.passed]

    if not relevant_results:
        return {
            "answer": "Retrieved content did not pass relevance verification.",
            "chunks": results,
            "relevance_checks": relevance_checks,
            "support_check": None,
        }

    context = "\n\n".join(r["text"] for r in relevant_results)
    answer = _generate(query, context)
    support_check = check_support(answer, [r["text"] for r in relevant_results])

    return {
        "answer": answer,
        "chunks": relevant_results,
        "relevance_checks": relevance_checks,
        "support_check": support_check,
    }


def answer_with_agentic(query: str) -> dict:
    """Multi-hop agentic retrieval (decompose -> retrieve -> observe) + generation + Self-RAG."""
    result = _agentic.run(query)

    if not result.final_chunks:
        return {
            "answer": "No relevant information found for any sub-question.",
            "hops": result.hops,
            "support_check": None,
        }

    chunk_texts = [c["payload"] for c in result.final_chunks]
    context = "\n\n".join(chunk_texts)
    answer = _generate(query, context)
    support_check = check_support(answer, chunk_texts)

    return {
        "answer": answer,
        "hops": result.hops,
        "support_check": support_check,
    }