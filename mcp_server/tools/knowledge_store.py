"""
KeywordStore: a lightweight BM25 keyword store for MarketLoop's RAG search
tool (search_marketloop_knowledge in rag_search.py).

This file was referenced by rag_indexing.py and rag_search.py
(from mcp_server.tools.knowledge_store import KeywordStore) but was never
committed to the repository - the import was broken for every teammate
pulling the branch. This is a from-scratch reconstruction matching the
exact interface both files already call:

  store = KeywordStore()
  store.upsert(payload=<str>, metadata=<dict>)
  store.query(query_text=<str>, top_k=<int>, filter=<dict>) -> list[dict]
      each result: {"score": float, "payload": str, "metadata": dict}

Implemented as pure-Python BM25 (no external dependency) so it doesn't
require adding a new package to requirements.txt for the rest of the team.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any
from RAG.metadata_index import MetadataIndex


_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class _Document:
    doc_id: int
    payload: str
    metadata: dict[str, Any]
    tokens: list[str] = field(default_factory=list)


def _idf(term: str, candidate_docs: list[_Document]) -> float:
    n = len(candidate_docs)
    n_containing = sum(1 for d in candidate_docs if term in d.tokens)
    if n_containing == 0:
        return 0.0
    return math.log((n - n_containing + 0.5) / (n_containing + 0.5) + 1)


def _matches_filter(metadata: dict[str, Any], filter: dict[str, Any]) -> bool:
    return all(metadata.get(k) == v for k, v in filter.items())


class KeywordStore:
    """BM25-ranked keyword store with metadata filtering."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: list[_Document] = []
        self._next_id = 0
        self._avg_doc_len = 0.0
        self.metadata_index = MetadataIndex()

    def upsert(self, payload: str, metadata: dict[str, Any] | None = None) -> int:
        """Add a document to the store. Returns its internal id."""
        doc = _Document(
            doc_id=self._next_id,
            payload=payload,
            metadata=metadata or {},
            tokens=_tokenize(payload),
        )

        self._docs.append(doc)
        self._next_id += 1
        self._recompute_avg_len()
        return doc.doc_id

    def _recompute_avg_len(self) -> None:
        if not self._docs:
            self._avg_doc_len = 0.0
            return
        self._avg_doc_len = sum(len(d.tokens) for d in self._docs) / len(self._docs)

    def _bm25_score(self, query_terms: list[str], doc: _Document, candidate_docs: list[_Document]) -> float:
        score = 0.0
        doc_len = len(doc.tokens)
        for term in query_terms:
            idf = _idf(term, candidate_docs)
            if idf == 0.0:
                continue
            term_freq = doc.tokens.count(term)
            numerator = term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (
                1 - self.b + self.b * (doc_len / (self._avg_doc_len or 1))
            )
            score += idf * (numerator / denominator)
        return score

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        filter = filter or {}
        candidate_ids = self.metadata_index.filter_ids(filter)

        if candidate_ids is None:
            candidates = self._docs
        else:
            candidates = [
                d for d in self._docs
                if d.doc_id in candidate_ids
            ]
        if not candidates:
            return []

        query_terms = _tokenize(query_text)
        scored = [
            (self._bm25_score(query_terms, d, candidates), d)
            for d in candidates
        ]
        scored = [(s, d) for s, d in scored if s > 0]
        scored.sort(key=lambda pair: pair[0], reverse=True)

        return [
            {"score": score, "payload": doc.payload, "metadata": doc.metadata}
            for score, doc in scored[:top_k]
        ]
