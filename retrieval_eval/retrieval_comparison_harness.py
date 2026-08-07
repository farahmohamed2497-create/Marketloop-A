"""
Retrieval architecture comparison (Concern: "Run every architecture
against every question and produce a comparison table across accuracy,
token usage, and latency. Then choose the architecture you'd actually
ship").

This harness currently runs Agentic RAG (this contributor's assigned
architecture) against the decomposition-required question set. Naive RAG
and hybrid search rows are marked "pending" until the teammates
responsible for those architectures plug their retrieve functions in -
same pattern used in context_eval/comparison_harness.py for zone-based
pruning.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

sys.path.append(str(BASE_DIR))


from RAG.agentic_rag import AgenticRAGRetriever
from RAG.self_rag_verification import check_relevance
from retrieval_eval.decomposition_questions import DECOMPOSITION_QUESTIONS

NAIVE_AVAILABLE = False   # flip once teammate's naive RAG pipeline lands
HYBRID_AVAILABLE = False  # flip once teammate's hybrid search lands


def count_tokens(text: str) -> int:
    return len(text.split())


def run_agentic(questions) -> dict:
    with open(BASE_DIR / "data" / "keyword_store.pkl", "rb") as f:
        store = pickle.load(f)

    retriever = AgenticRAGRetriever(
        store,
        top_k=2
    )

    correct = 0
    total_tokens = 0
    start = time.perf_counter()

    for q in questions:
        result = retriever.run(q["query"])

        covered_subsections = {
            r["metadata"].get("subsection")
            for hop in result.hops
            for r in hop.results
        }
        hit = q["expected_subsections"].issubset(covered_subsections)

        # only count chunks that pass the relevance check toward token cost,
        # matching what a Self-RAG-filtered pipeline would actually send on
        relevant_chunks = [
            r["payload"] for hop in result.hops for r in hop.results
            if check_relevance(hop.query, r["payload"]).passed
        ]
        total_tokens += sum(count_tokens(c) for c in relevant_chunks)

        if hit:
            correct += 1

    elapsed = time.perf_counter() - start
    n = len(questions)

    return {
        "architecture": "Agentic RAG (multi-hop)",
        "accuracy": f"{correct}/{n}",
        "avg_tokens": round(total_tokens / n, 1),
        "avg_latency_ms": round(1000 * elapsed / n, 3),
    }


def main():
    rows = [run_agentic(DECOMPOSITION_QUESTIONS)]

    if not NAIVE_AVAILABLE:
        rows.insert(0, {"architecture": "Naive RAG", "accuracy": "pending", "avg_tokens": "-", "avg_latency_ms": "-"})
    if not HYBRID_AVAILABLE:
        rows.insert(1, {"architecture": "Hybrid search", "accuracy": "pending", "avg_tokens": "-", "avg_latency_ms": "-"})

    print(f"\nTest set: {len(DECOMPOSITION_QUESTIONS)} decomposition-required questions\n")
    print("| Architecture | Accuracy | Avg tokens/query | Avg latency (ms) |")
    print("|---|---|---|---|")
    for r in rows:
        print(f"| {r['architecture']} | {r['accuracy']} | {r['avg_tokens']} | {r['avg_latency_ms']} |")

    return rows


if __name__ == "__main__":
    main()