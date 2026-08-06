"""
Decomposition-required test questions (Concern: "a domain-specific test
set... something that needs decomposition or multiple retrieval rounds
that only agentic RAG can handle well").

Each question references two genuinely separate indexed documents
(different `subsection` metadata), grounded in the real content indexed
by mcp_server/tools/rag_indexing.py - not invented text. A single BM25
query tends to surface only one side, which is exactly the failure mode
agentic RAG's decomposition step is meant to fix.
"""

from __future__ import annotations

DECOMPOSITION_QUESTIONS = [
    {
        "query": "What's the return policy for a defective item, and which tool processes an approved return?",
        "expected_subsections": {"Returns", "Customer Service"},
    },
    {
        "query": "What are the inventory reorder rules, and who is authorized to update inventory?",
        "expected_subsections": {"Inventory Management", "Access Control"},
    },
    {
        "query": "How does order fulfillment work after payment is confirmed, and what gets logged to the audit log during that process?",
        "expected_subsections": {"Order Fulfillment", "Compliance"},
    },
]
