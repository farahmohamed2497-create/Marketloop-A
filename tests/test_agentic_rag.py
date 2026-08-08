import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from mcp_server.tools.knowledge_store import KeywordStore
from RAG.chunking import chunk_document
from RAG.agentic_rag import AgenticRAGRetriever


CATALOG_PATH = Path(__file__).resolve().parents[1] / "Complete Enterprise Product Catalog.txt"


def _make_store() -> KeywordStore:
    store = KeywordStore()
    text = CATALOG_PATH.read_text(encoding="utf-8")

    for chunk in chunk_document(
        text,
        base_metadata={"doc": "Complete Enterprise Product Catalog"},
    ):
        store.upsert(
            payload=chunk["text"],
            metadata=chunk["metadata"],
        )

    return store


def test_single_topic_query_is_not_decomposed():
    retriever = AgenticRAGRetriever(_make_store(), top_k=3)
    result = retriever.run("What is the return policy?")

    assert result.hop_count == 1
    assert "not needed" in result.hops[0].reasoning.lower() or "single-topic" in result.hops[0].reasoning.lower()


def test_multi_topic_query_is_decomposed_into_multiple_hops():
    retriever = AgenticRAGRetriever(_make_store(), top_k=2)
    result = retriever.run(
        "What's the return policy for a defective item, and which tool processes an approved return?"
    )

    assert result.hop_count >= 2


def test_decomposition_covers_both_expected_topics():
    retriever = AgenticRAGRetriever(_make_store(), top_k=2)
    result = retriever.run(
        "What's the return eligibility for the UltraView 4K Smart TV, "
        "and how long does standard shipping delivery take?"
    )

    combined_text = " ".join(
        result_item["payload"]
        for hop in result.hops
        for result_item in hop.results
    )

    assert "UltraView" in combined_text
    assert "Standard Delivery" in combined_text


def test_max_hops_is_respected():
    retriever = AgenticRAGRetriever(_make_store(), top_k=1, max_hops=1)
    result = retriever.run(
        "What's the return policy for a defective item, and which tool processes an approved return, "
        "and what gets logged to the audit log, and who is authorized to update inventory?"
    )

    assert result.hop_count <= 1


def test_reasoning_is_logged_for_every_hop():
    retriever = AgenticRAGRetriever(_make_store(), top_k=2)
    result = retriever.run("What is the return policy for a defective item?")

    assert all(hop.reasoning for hop in result.hops)
