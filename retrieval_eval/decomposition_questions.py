"""
Fixed multi-part questions for Agentic RAG evaluation.

Each question requires evidence from two separate parts of the real
MarketLoop catalog. The expected keyword groups are checked only against
retrieved chunks, so every architecture can be evaluated on the same
evidence-based metric.
"""

from __future__ import annotations


DECOMPOSITION_QUESTIONS = [
    {
        "id": "multi-return-shipping",
        "category": "multi_part",
        "query": (
            "What is the return eligibility for the UltraView 4K Smart TV, "
            "and how long does standard delivery take?"
        ),
        "expected_keyword_groups": [
            ["UltraView"],
            ["Return Eligibility"],
            ["Standard Delivery"],
        ],
    },
    {
        "id": "multi-price-promo",
        "category": "multi_part",
        "query": (
            "What is the price of the BrewMaster Digital Espresso Machine, "
            "and what discount does PROMO-CODE: SAVE10 provide?"
        ),
        "expected_keyword_groups": [
            ["BrewMaster"],
            ["$299.99"],
            ["SAVE10"],
            ["10%"],
        ],
    },
    {
        "id": "multi-warranty-shipping",
        "category": "multi_part",
        "query": (
            "What warranty applies to SKU SPOR-303, "
            "and how long does standard delivery take?"
        ),
        "expected_keyword_groups": [
            ["SPOR-303"],
            ["5-Year Frame Warranty"],
            ["Standard Delivery"],
        ],
    },
]