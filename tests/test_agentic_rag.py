import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from mcp_server.tools.knowledge_store import KeywordStore
from mcp_server.tools.rag_indexing import index_marketloop_knowledge
from rag.agentic_rag import AgenticRAGRetriever


def _make_store() -> KeywordStore:
    store = KeywordStore()
    index_marketloop_knowledge(store)
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
        "What are the inventory reorder rules, and who is authorized to update inventory?"
    )

    covered = {
        r["metadata"].get("subsection")
        for hop in result.hops for r in hop.results
    }
    assert {"Inventory Management", "Access Control"}.issubset(covered)


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
