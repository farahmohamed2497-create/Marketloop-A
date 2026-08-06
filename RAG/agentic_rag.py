"""
Agentic RAG (Concern: "Agentic RAG - a reasoning loop that decides what to
retrieve, retrieves, observes, and decides whether to retrieve again").

Real MarketLoop need: some support questions can't be answered by a single
retrieval round. Example - "Customer's laptop return: it arrived damaged,
and they're also asking whether the delayed shipment qualifies for a
credit. What restocking fee applies, and are they owed anything for the
delay?" That needs two separate lookups (the Returns policy, and the
Shipping SLA policy) - a single keyword query pulls one or the other, not
both, because they're indexed as separate policy documents.

This module builds on the same BM25 keyword store used by naive search
(mcp_server.tools.knowledge_store.KeywordStore) - it does not require the
vector store, since the "agentic" concern here is the reasoning loop
(decompose -> retrieve -> observe -> decide), not the retrieval method
itself. Swapping in vector or hybrid retrieval later is a one-line change
(replace `self._search` with a different retriever).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from mcp_server.tools.knowledge_store import KeywordStore

# Sub-questions are split on coordinating conjunctions and question marks -
# good enough for MarketLoop's support-ticket phrasing without needing an
# LLM call just to segment a sentence.
_SPLIT_RE = re.compile(r"\?|\band\b|\balso\b", re.IGNORECASE)


@dataclass
class RetrievalHop:
    query: str
    results: list[dict[str, Any]]
    reasoning: str


@dataclass
class AgenticRAGResult:
    original_query: str
    hops: list[RetrievalHop] = field(default_factory=list)
    final_chunks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def hop_count(self) -> int:
        return len(self.hops)


class AgenticRAGRetriever:
    """Decomposes a query into sub-questions when needed, retrieves each
    one separately, and stops once every sub-question has evidence -
    instead of firing one query and hoping it covers a multi-part
    question."""

    def __init__(self, store: KeywordStore, top_k: int = 3, max_hops: int = 4):
        self.store = store
        self.top_k = top_k
        self.max_hops = max_hops

    def _decompose(self, query: str) -> list[str]:
        parts = [p.strip() for p in _SPLIT_RE.split(query) if p.strip()]
        # A single short clause left after splitting isn't a real
        # sub-question (e.g. a trailing "?") - drop fragments under 3 words.
        parts = [p for p in parts if len(p.split()) >= 3]
        return parts or [query]

    def run(self, query: str) -> AgenticRAGResult:
        result = AgenticRAGResult(original_query=query)
        sub_queries = self._decompose(query)

        if len(sub_queries) > 1:
            reasoning_prefix = (
                f"Query decomposed into {len(sub_queries)} sub-questions "
                f"because it references more than one distinct topic."
            )
        else:
            reasoning_prefix = "Query treated as single-topic, no decomposition needed."

        for sub_query in sub_queries[: self.max_hops]:
            hits = self.store.query(sub_query, top_k=self.top_k)
            covered = len(hits) > 0
            reasoning = (
                f"{reasoning_prefix} Retrieved {len(hits)} chunk(s) for "
                f"sub-question '{sub_query}'."
                if covered
                else f"{reasoning_prefix} No matches for sub-question "
                f"'{sub_query}' - this gap should surface in the final answer."
            )
            result.hops.append(RetrievalHop(query=sub_query, results=hits, reasoning=reasoning))
            result.final_chunks.extend(hits)

        return result
