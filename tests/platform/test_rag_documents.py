"""e2e: admin adds/removes a RAG document through the platform's HTTP API,
and a (simulated) agent process picks up the change on its next query
without a restart.

The embedding model and the Groq generation call are faked so this test
runs offline and fast -- what's under test is the wiring (admin API ->
document_store -> vector_store/keyword_store -> rag_pipeline's version
check), not sentence-transformers or an LLM.
"""
from __future__ import annotations

import numpy as np
import pytest

import rag.document_store as document_store_module
import rag.rag_pipeline as rag_pipeline


class _FakeEmbeddingModel:
    """Deterministic, dependency-free stand-in for sentence-transformers.

    A tiny bag-of-words hash embedding: shared vocabulary between a query
    and a chunk pulls their vectors together under cosine similarity,
    same as a real embedding model would, without downloading one.
    """

    dim = 16

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for word in text.lower().split():
            vec[hash(word) % self.dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed(self, text: str) -> np.ndarray:
        return self._vector(text)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vector(t) for t in texts])


@pytest.fixture(autouse=True)
def fake_embedder(monkeypatch, tmp_path):
    """Route document_store + rag_pipeline's on-disk store paths into the
    same tmp_path/data layout rag_pipeline expects, and swap in the fake
    embedder everywhere one is built."""
    data_dir = tmp_path / "data"
    monkeypatch.setattr(document_store_module, "VECTOR_DB_PATH", data_dir / "marketloop_vector_db")
    monkeypatch.setattr(document_store_module, "KEYWORD_STORE_PATH", data_dir / "keyword_store.pkl")
    monkeypatch.setattr(document_store_module, "EmbeddingModel", _FakeEmbeddingModel)

    monkeypatch.setattr(rag_pipeline, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(rag_pipeline, "EmbeddingModel", _FakeEmbeddingModel)
    monkeypatch.setattr(rag_pipeline, "_generate", lambda query, context: f"ANSWER using: {context}")
    monkeypatch.setattr(
        rag_pipeline,
        "check_relevance",
        lambda query, text: type("R", (), {"passed": True})(),
    )
    monkeypatch.setattr(
        rag_pipeline,
        "check_support",
        lambda answer, chunks: type("R", (), {"passed": True})(),
    )

    # Force a clean in-process cache for every test.
    monkeypatch.setattr(rag_pipeline, "_embedder", None)
    monkeypatch.setattr(rag_pipeline, "_naive", None)
    monkeypatch.setattr(rag_pipeline, "_hybrid", None)
    monkeypatch.setattr(rag_pipeline, "_agentic", None)
    monkeypatch.setattr(rag_pipeline, "_loaded_store_version", None)


def test_added_document_is_retrieved_on_next_agent_query(client):
    response = client.post(
        "/api/rag/documents",
        json={
            "filename": "warp_drive_manual.txt",
            "text": "The warp-drive-manual explains how to recalibrate the dilithium matrix safely.",
        },
    )
    assert response.status_code == 201
    document_id = response.json()["document_id"]

    listed = client.get("/api/rag/documents").json()
    assert any(doc["document_id"] == document_id for doc in listed)

    # Simulate an already-running agent process's next query: it hasn't
    # loaded anything yet in this test, so this is really exercising the
    # "reflected in what that agent retrieves on the next query" path.
    result = rag_pipeline.answer_with_naive("How do I recalibrate the dilithium matrix?", top_k=1)

    assert result["chunks"], "expected the newly-added document to be retrievable"
    assert "warp-drive-manual" in result["chunks"][0]["text"]


def test_removed_document_stops_being_retrieved(client):
    add_response = client.post(
        "/api/rag/documents",
        json={
            "filename": "warp_drive_manual.txt",
            "text": "The warp-drive-manual explains how to recalibrate the dilithium matrix safely.",
        },
    )
    document_id = add_response.json()["document_id"]

    # Prime the in-process cache the way a long-running agent would have
    # it warm before the admin ever touches the store.
    primed = rag_pipeline.answer_with_naive("dilithium matrix", top_k=1)
    assert primed["chunks"], "sanity check: document should be retrievable before removal"

    delete_response = client.delete(f"/api/rag/documents/{document_id}")
    assert delete_response.status_code == 200
    assert delete_response.json()["removed_chunks"] >= 1

    assert client.get("/api/rag/documents").json() == []

    # Same already-warm process, next query: must NOT still see the
    # removed document just because it was cached in memory.
    result = rag_pipeline.answer_with_naive("dilithium matrix", top_k=1)
    assert result["chunks"] == [] or "warp-drive-manual" not in result["chunks"][0]["text"]


def test_removing_unknown_document_is_a_404(client):
    response = client.delete("/api/rag/documents/does-not-exist")
    assert response.status_code == 404
