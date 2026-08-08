"""
General single-topic questions for the Naive RAG baseline.

These questions do not require exact identifiers or multi-hop reasoning.
They establish the simple vector-retrieval baseline used in the final
comparison table.
"""

from __future__ import annotations


GENERAL_QUESTIONS = [
    {
        "id": "general-warranty",
        "category": "general",
        "query": "What warranty does the UltraView 4K Smart TV include?",
        "expected_keyword_groups": [
            ["UltraView"],
            ["2-Year Comprehensive"],
        ],
    },
    {
        "id": "general-shipping",
        "category": "general",
        "query": "How long does standard delivery take?",
        "expected_keyword_groups": [
            ["Standard Delivery"],
            ["3 to 5 business days"],
        ],
    },
    {
        "id": "general-defect-return",
        "category": "general",
        "query": "What happens when an item has shipping damage reported within 48 hours?",
        "expected_keyword_groups": [
            ["shipping damage"],
            ["15% restocking fee is entirely waived"],
        ],
    },
]