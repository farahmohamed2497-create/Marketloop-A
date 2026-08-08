"""Agentic RAG for multi-part MarketLoop support questions.

The retriever follows an explicit retrieve -> observe -> decide loop.  It
can use the local BM25 KeywordStore today, or an injected Hybrid-search
adapter once the shared hybrid pipeline is wired by the integration owner.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from mcp_server.tools.knowledge_store import KeywordStore


SearchFn = Callable[[str, int], list[dict[str, Any]]]
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
    """Retrieve evidence iteratively for each distinct part of a question.

    ``search_fn`` is intentionally injectable.  The current unit-tested
    fallback is ``KeywordStore.query``; the production integration can pass
    a Hybrid Search adapter with the same ``(query, top_k)`` signature.
    """

    def __init__(
        self,
        store: KeywordStore | None = None,
        *,
        search_fn: SearchFn | None = None,
        top_k: int = 3,
        max_hops: int = 4,
    ) -> None:
        if store is None and search_fn is None:
            raise ValueError("Provide a KeywordStore or a search_fn.")

        self.store = store
        self.search_fn = search_fn
        self.top_k = top_k
        self.max_hops = max_hops

    def _decompose(self, query: str) -> list[str]:
        parts = [part.strip() for part in _SPLIT_RE.split(query) if part.strip()]
        parts = [part for part in parts if len(part.split()) >= 3]
        return parts or [query]

    def _search(self, query: str) -> list[dict[str, Any]]:
        """Run the configured retriever and normalize its result shape."""
        if self.search_fn is not None:
            raw_results = self.search_fn(query, self.top_k)
        else:
            assert self.store is not None
            raw_results = self.store.query(query, top_k=self.top_k)

        normalized: list[dict[str, Any]] = []
        for result in raw_results:
            # KeywordStore returns payload; HybridSearch returns text.
            payload = result.get("payload", result.get("text", ""))
            normalized.append(
                {
                    "payload": payload,
                    "metadata": result.get("metadata", {}),
                    "score": result.get("score", 0.0),
                }
            )
        return normalized

    @staticmethod
    def _fallback_query(sub_query: str, original_query: str) -> str | None:
        """Choose one broader retry when a narrow sub-question has no evidence."""
        if sub_query.strip().lower() != original_query.strip().lower():
            return original_query
        return None

    def run(self, query: str) -> AgenticRAGResult:
        result = AgenticRAGResult(original_query=query)
        planned_queries = self._decompose(query)
        plan_note = (
            f"Query decomposed into {len(planned_queries)} sub-questions."
            if len(planned_queries) > 1
            else "Query treated as single-topic; decomposition not needed."
        )
        pending: deque[str] = deque(planned_queries)
        attempted: set[str] = set()

        while pending and len(result.hops) < self.max_hops:
            sub_query = pending.popleft()
            normalized_query = sub_query.casefold()
            if normalized_query in attempted:
                continue
            attempted.add(normalized_query)

            hits = self._search(sub_query)
            result.final_chunks.extend(hits)

            if hits:
                reasoning = (
                    f"{plan_note} Retrieved {len(hits)} evidence chunk(s) for '{sub_query}'. "
                    "Observation: evidence found, so no retry is needed for this sub-question."
                )
            else:
                fallback = self._fallback_query(sub_query, query)
                if fallback and fallback.casefold() not in attempted and len(result.hops) + len(pending) < self.max_hops:
                    pending.append(fallback)
                    reasoning = (
                        f"{plan_note} No evidence found for '{sub_query}'. Observation: coverage is insufficient; "
                        f"retrieve again with broader query '{fallback}'."
                    )
                else:
                    reasoning = (
                        f"{plan_note} No evidence found for '{sub_query}'. Observation: no safe additional query "
                        "is available within the hop limit, so surface the gap in the final answer."
                    )

            result.hops.append(RetrievalHop(query=sub_query, results=hits, reasoning=reasoning))

        return result