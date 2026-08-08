"""
Single fixed retrieval evaluation set.

Every architecture must run against this exact list:
- general questions favor the Naive baseline;
- identifier questions expose Hybrid Search's BM25 advantage;
- multi-part questions expose Agentic RAG's multi-hop advantage.
"""

from __future__ import annotations

import json
from pathlib import Path

from retrieval_eval.decomposition_questions import DECOMPOSITION_QUESTIONS
from retrieval_eval.general_questions import GENERAL_QUESTIONS


_BASE_DIR = Path(__file__).resolve().parent

with (_BASE_DIR / "citation_questions.json").open(encoding="utf-8") as file:
    IDENTIFIER_QUESTIONS = json.load(file)

RETRIEVAL_QUESTIONS = (
    GENERAL_QUESTIONS
    + IDENTIFIER_QUESTIONS
    + DECOMPOSITION_QUESTIONS
)


def expected_answer_is_retrieved(question: dict, chunks: list[str]) -> bool:
    """Return True only when every expected evidence group appears in chunks."""
    combined_text = " ".join(chunks)

    return all(
        any(keyword in combined_text for keyword in keyword_group)
        for keyword_group in question["expected_keyword_groups"]
    )