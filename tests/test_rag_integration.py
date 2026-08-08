"""
Integration tests for the RAG flow end-to-end.
Covers: hybrid retrieval -> generation -> Self-RAG checks, and
agentic retrieval -> generation -> Self-RAG checks.

Uses fake embedder/store/LLM client (no real API calls, no real disk
files needed) so tests stay fast, deterministic, and don't require
GROQ_API_KEY or a pre-built vector/keyword store on disk.
"""

from types import SimpleNamespace

import pytest

import RAG.rag_pipeline as pipeline
from RAG.agentic_rag import RetrievalHop, AgenticRAGResult


# ---------- Fakes ----------

class FakeEmbedder:
    def embed(self, text):
        return [0.1, 0.2, 0.3]  # القيمة مش مهمة، الـ FakeHybrid مش هيستخدمها فعليًا


class FakeHybrid:
    def __init__(self, results):
        self._results = results

    def search(self, query_embedding, query_text, top_k=3):
        return self._results


class FakeAgenticRetriever:
    def __init__(self, result: AgenticRAGResult):
        self._result = result

    def run(self, query):
        return self._result


class FakeNaiveRetriever:
    def __init__(self, results):
        self._results = results

    def retrieve(self, query, *, top_k=3, filters=None):
        return self._results


class FakeGroqCompletions:
    def __init__(self, answer_text: str):
        self._answer_text = answer_text

    def create(self, model, messages, temperature):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._answer_text))]
        )


class FakeGroqClient:
    def __init__(self, answer_text: str):
        self.chat = SimpleNamespace(completions=FakeGroqCompletions(answer_text))


# ---------- Hybrid RAG: happy path ----------

def test_answer_with_hybrid_returns_grounded_answer(monkeypatch):
    fake_results = [
        {
            "text": "Return policy: items can be returned within 14 days of delivery.",
            "metadata": {"doc": "returns"},
            "score": 0.9,
        }
    ]
    monkeypatch.setattr(pipeline, "_embedder", FakeEmbedder())
    monkeypatch.setattr(pipeline, "_hybrid", FakeHybrid(fake_results))
    monkeypatch.setattr(
        pipeline, "_groq_client",
        FakeGroqClient("You can return items within 14 days of delivery."),
    )

    result = pipeline.answer_with_hybrid("What is the return policy?")

    assert "14 days" in result["answer"]
    assert len(result["chunks"]) == 1
    assert result["relevance_checks"][0].passed  # الكلمات متطابقة (return, policy)
    assert result["support_check"].passed  # "14 days" موجودة في الـ chunk المسترجع


# ---------- Hybrid RAG: no results ----------

def test_answer_with_hybrid_handles_empty_results(monkeypatch):
    monkeypatch.setattr(pipeline, "_embedder", FakeEmbedder())
    monkeypatch.setattr(pipeline, "_hybrid", FakeHybrid([]))
    monkeypatch.setattr(pipeline, "_groq_client", FakeGroqClient("should not be called"))

    result = pipeline.answer_with_hybrid("Completely unrelated question about the moon")

    assert result["answer"] == "No relevant information found in the knowledge base."
    assert result["chunks"] == []
    assert result["support_check"] is None


# ---------- Hybrid RAG: Self-RAG catches an irrelevant chunk ----------

def test_answer_with_hybrid_filters_out_irrelevant_chunk(monkeypatch):
    """
    السيناريو المطلوب في الرابريك: chunk سطحيًا مش متعلق بالسؤال
    (مفيش تداخل كلمات معنوي كافي) لازم يترفض قبل ما يدخل الـ context.
    """
    fake_results = [
        {
            "text": "The warehouse cat likes to sleep on shipping boxes all day.",
            "metadata": {"doc": "random"},
            "score": 0.5,
        }
    ]
    monkeypatch.setattr(pipeline, "_embedder", FakeEmbedder())
    monkeypatch.setattr(pipeline, "_hybrid", FakeHybrid(fake_results))
    monkeypatch.setattr(pipeline, "_groq_client", FakeGroqClient("should not be called"))

    result = pipeline.answer_with_hybrid("What is the restocking fee for electronics?")

    assert result["answer"] == "Retrieved content did not pass relevance verification."
    assert result["relevance_checks"][0].passed is False


# ---------- Agentic RAG: happy path with multiple hops ----------

def test_answer_with_agentic_returns_grounded_answer_with_hops(monkeypatch):
    fake_agentic_result = AgenticRAGResult(
        original_query="What is the restocking fee, and are they owed a delay credit?",
        hops=[
            RetrievalHop(
                query="What is the restocking fee",
                results=[{"payload": "Restocking fee is 10% for opened electronics.", "metadata": {}, "score": 1.0}],
                reasoning="Retrieved 1 chunk for restocking fee sub-question.",
            ),
            RetrievalHop(
                query="are they owed a delay credit",
                results=[{"payload": "Delayed shipments over 5 days qualify for a $10 credit.", "metadata": {}, "score": 1.0}],
                reasoning="Retrieved 1 chunk for delay credit sub-question.",
            ),
        ],
        final_chunks=[
            {"payload": "Restocking fee is 10% for opened electronics.", "metadata": {}, "score": 1.0},
            {"payload": "Delayed shipments over 5 days qualify for a $10 credit.", "metadata": {}, "score": 1.0},
        ],
    )

    monkeypatch.setattr(pipeline, "_agentic", FakeAgenticRetriever(fake_agentic_result))
    monkeypatch.setattr(
        pipeline, "_groq_client",
        FakeGroqClient("The restocking fee is 10%, and they qualify for a $10 delay credit."),
    )

    result = pipeline.answer_with_agentic(
        "What is the restocking fee, and are they owed a delay credit?"
    )

    assert "10%" in result["answer"]
    assert "$10" in result["answer"]
    assert len(result["hops"]) == 2  # اتعمل decomposition فعلي، مش استرجاع واحد
    assert result["support_check"].passed


# ---------- Agentic RAG: no chunks found for any sub-question ----------

def test_answer_with_agentic_handles_no_chunks(monkeypatch):
    fake_agentic_result = AgenticRAGResult(
        original_query="unrelated multi-part question",
        hops=[],
        final_chunks=[],
    )

    monkeypatch.setattr(pipeline, "_agentic", FakeAgenticRetriever(fake_agentic_result))
    monkeypatch.setattr(pipeline, "_groq_client", FakeGroqClient("should not be called"))

    result = pipeline.answer_with_agentic("unrelated multi-part question")

    assert result["answer"] == "No relevant information found for any sub-question."
    assert result["support_check"] is None


# ---------- Self-RAG catches an ungrounded (fabricated) answer ----------

def test_support_check_flags_fabricated_answer(monkeypatch):
    """
    السيناريو المطلوب في الرابريك: الإجابة المولّدة فيها ادعاءات
    مش موجودة في الـ chunks المسترجعة أصلًا (هلوسة محتملة).
    """
    fake_results = [
        {
            "text": "Return policy: items can be returned within 14 days.",
            "metadata": {"doc": "returns"},
            "score": 0.9,
        }
    ]
    monkeypatch.setattr(pipeline, "_embedder", FakeEmbedder())
    monkeypatch.setattr(pipeline, "_hybrid", FakeHybrid(fake_results))
    # الإجابة بتدّعي رقم مش موجود في الـ chunk خالص (30 يوم، مش 14)
    monkeypatch.setattr(
        pipeline, "_groq_client",
        FakeGroqClient("You get a full refund within 30 days and free express shipping."),
    )

    result = pipeline.answer_with_hybrid("What is the return policy?")

    assert result["support_check"].passed is False
    assert result["answer"].startswith(
        "I can't provide a grounded answer"
    )

def test_answer_with_naive_returns_grounded_answer(monkeypatch):
    fake_results = [
        {
            "text": "SKU ELEC-001 includes a two-year manufacturer warranty.",
            "metadata": {"doc": "catalog"},
            "score": 0.9,
        }
    ]

    monkeypatch.setattr(pipeline, "_naive", FakeNaiveRetriever(fake_results))
    monkeypatch.setattr(
        pipeline,
        "_groq_client",
        FakeGroqClient("SKU ELEC-001 includes a two-year manufacturer warranty."),
    )

    result = pipeline.answer_with_naive("What warranty applies to SKU ELEC-001?")

    assert "two-year" in result["answer"]
    assert len(result["chunks"]) == 1
    assert result["relevance_checks"][0].passed
    assert result["support_check"].passed