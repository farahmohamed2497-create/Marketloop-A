"""
decompose_and_search: breaks a compound question into 2-4 sub-questions
via one LLM call, searches each sub-question separately using
search_knowledge_base, and returns all results tagged with which
sub-question they answer. Sits in front of search_knowledge_base —
does not replace it.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ---------------------------------------------------------------------------
# Knowledge base: loaded directly from the catalog text file, self-contained
# ---------------------------------------------------------------------------

_CATALOG_PATH = Path(__file__).resolve().parents[2] / "Complete Enterprise Product Catalog.txt"

with open(_CATALOG_PATH, encoding="utf-8") as f:
    _catalog_text = f.read()

# Split the catalog into simple chunks (by blank lines / paragraphs)
_CHUNKS = [
    chunk.strip()
    for chunk in re.split(r"\n\s*\n", _catalog_text)
    if len(chunk.strip()) > 20
]


def _keywords(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def search_knowledge_base(query: str, top_k: int = 3):
    """
    The existing search tool: keyword-overlap search over the MarketLoop
    knowledge base (product catalog + policies). Returns a list of
    (chunk_text, score) tuples, best matches first.
    """
    query_kw = _keywords(query)
    scored = []
    for chunk in _CHUNKS:
        chunk_kw = _keywords(chunk)
        overlap = len(query_kw & chunk_kw)
        if overlap > 0:
            scored.append((chunk, overlap / len(query_kw)))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# LLM: one call to split a compound query into sub-questions
# ---------------------------------------------------------------------------

DECOMPOSE_PROMPT = """\
Break the following question into 2-4 simpler sub-questions that, together,
fully answer it. If the question is already simple, just return it as-is
as a single sub-question.

Question: {query}

Return ONLY a numbered list, one sub-question per line. Example:
1. ...
2. ...
"""


def decompose_query(query: str) -> list[str]:
    """Turn one (possibly compound) query into a list of sub-questions."""
    response = _groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": DECOMPOSE_PROMPT.format(query=query)}],
        temperature=0,
    )
    raw = response.choices[0].message.content

    sub_questions = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        for sep in [". ", ") ", "- "]:
            if sep in line[:4]:
                line = line.split(sep, 1)[1]
                break
        sub_questions.append(line.strip())

    return sub_questions or [query]


# ---------------------------------------------------------------------------
# The new tool: decompose_and_search
# ---------------------------------------------------------------------------

@dataclass
class TaggedChunk:
    sub_question: str
    chunk: str
    score: float


def decompose_and_search(query: str, top_k: int = 3):
    """
    MCP tool: breaks a compound question into sub-questions via LLM,
    searches each one via search_knowledge_base, and returns all chunks
    tagged with which sub-question they came from. Does not merge them
    into a final answer — that's left to the model.
    """
    sub_questions = decompose_query(query)

    results: list[dict] = []
    for sub_q in sub_questions:
        hits = search_knowledge_base(sub_q, top_k)
        for chunk, score in hits:
            results.append({
                "sub_question": sub_q,
                "chunk": chunk,
                "score": score,
            })

    return results


# ---------------------------------------------------------------------------
# Demo: compound question that plain search answers only partially
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_query = "What is the restocking fee for electronics, and how long is the standard shipping delivery window?"

    print("=== Plain search (sees the raw compound question only) ===")
    plain_hits = search_knowledge_base(demo_query, top_k=3)
    for chunk, score in plain_hits:
        print(f"  (score={score:.2f}) {chunk[:100]}")

    print("\n=== decompose_and_search ===")
    tagged_results = decompose_and_search(demo_query, top_k=2)
    for r in tagged_results:
        print(f"  [{r['sub_question']}]")
        print(f"    -> (score={r['score']:.2f}) {r['chunk'][:100]}")