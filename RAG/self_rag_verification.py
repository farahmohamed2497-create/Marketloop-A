"""
Self-RAG-style verification (Concern: "an explicit check, is the
retrieved content actually relevant, is the generated answer actually
supported by it... applies to both RAG answers and to memories recalled
from the episodic and semantic store").

Real MarketLoop need: a retrieved policy chunk that superficially matches
keywords but isn't actually about the customer's situation (e.g. a
"Shipping SLA" chunk surfacing for a restocking-fee question because both
mention "return") would let the agent state a fee rule it never actually
checked. Same risk on the memory side: a promoted episodic memory (e.g.
"customer's return reason: damaged in shipping") should not get reused for
an unrelated new case just because it shares generic words.

No LLM call is used here - both checks are lexical-overlap heuristics,
consistent with the rest of this repo's memory/RAG modules (masking,
summarization, agentic decomposition), which are all rule-based rather
than LLM-based to keep this fully unit-testable and free of API cost.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "in",
    "on", "for", "and", "or", "this", "that", "it", "as", "with", "at",
    "by", "from", "what", "does", "do", "did",
}


def _keywords(text: str) -> set[str]:
    return {
        w.lower() for w in _WORD_RE.findall(text)
        if w.lower() not in _STOPWORDS and len(w) > 2
    }


@dataclass
class VerificationResult:
    passed: bool
    overlap_ratio: float
    reasoning: str


def check_relevance(query: str, chunk_text: str, threshold: float = 0.15) -> VerificationResult:
    """Is this retrieved chunk actually about what the query asked, or did
    it just share a stray keyword? Ratio of query keywords found in the
    chunk, against a minimum threshold."""
    query_kw = _keywords(query)
    chunk_kw = _keywords(chunk_text)

    if not query_kw:
        return VerificationResult(False, 0.0, "Query had no meaningful keywords to check against.")

    overlap = len(query_kw & chunk_kw) / len(query_kw)
    passed = overlap >= threshold

    reasoning = (
        f"{overlap:.0%} of query keywords found in chunk "
        f"({'meets' if passed else 'below'} {threshold:.0%} threshold)."
    )
    return VerificationResult(passed, overlap, reasoning)


def check_support(answer: str, chunks: list[str], threshold: float = 0.2) -> VerificationResult:
    """Is the generated answer actually grounded in the retrieved chunks,
    or does it contain claims that never appeared in what was retrieved?
    Ratio of answer keywords traceable to the combined chunk text."""
    answer_kw = _keywords(answer)
    combined_kw: set[str] = set()
    for c in chunks:
        combined_kw |= _keywords(c)

    if not answer_kw:
        return VerificationResult(False, 0.0, "Answer had no meaningful keywords to check.")
    if not chunks:
        return VerificationResult(False, 0.0, "No retrieved chunks to support any answer.")

    overlap = len(answer_kw & combined_kw) / len(answer_kw)
    passed = overlap >= threshold

    reasoning = (
        f"{overlap:.0%} of answer keywords are traceable to retrieved chunks "
        f"({'grounded' if passed else 'UNGROUNDED - answer may be fabricated'})."
    )
    return VerificationResult(passed, overlap, reasoning)


def check_memory_recall(current_context: str, recalled_content: str, threshold: float = 0.15) -> VerificationResult:
    """Same relevance check, applied to a memory item recalled from the
    episodic/semantic store instead of a RAG chunk - a promoted memory
    (e.g. an EpisodicMemory.content or a semantic fact) should only be
    reused if it's actually relevant to the current conversation."""
    return check_relevance(current_context, recalled_content, threshold=threshold)
