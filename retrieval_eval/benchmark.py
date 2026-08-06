import time
from pathlib import Path
import json
from RAG.vector_store import VectorStore
from RAG.embedding import EmbeddingModel
from RAG.hybrid_search import HybridSearch
from mcp_server.tools.knowledge_store import KeywordStore

def evaluate(name, search_fn, question):

    hits = 0
    total_latency = 0

    for q in question:

        start = time.perf_counter()

        result = search_fn(q["question"])

        latency = time.perf_counter() - start

        total_latency += latency

        combined_text = " ".join(
            str(re)
            for re in result
        ).lower()

        if q["answer_contains"].lower() in combined_text:
            hits += 1

    return {
        "system": name,
        "accuracy": hits / len(question),
        "latency_ms": (
            total_latency /
            len(question)
        ) * 1000
    }


BASE_DIR = Path(__file__).parent

with open(
    BASE_DIR / "citation_questions.json",
    encoding="utf8"
) as f:
    questions = json.load(f)


print("Benchmark Ready")

embedder = EmbeddingModel()

vector_store = VectorStore.load(
    "./data/marketloop_vector_db"
)

keyword_store = KeywordStore()

def vector_search(question):

    query_embedding = embedder.embed(question)

    return vector_store.search(
        query_embedding,
        k=5
    )

def bm25_search(question):

    return keyword_store.query(
        question,
        top_k=5
    )

hybrid = HybridSearch(
    vector_store,
    keyword_store
)

def hybrid_search(question):

    query_embedding = embedder.embed(
        question
    )

    return hybrid.search(
        query_embedding=query_embedding,
        query_text=question,
        top_k=5
    )

results = [
    evaluate(
        "Vector",
        vector_search,
        questions
    ),
    evaluate(
        "BM25",
        bm25_search,
        questions
    ),
    evaluate(
        "Hybrid",
        hybrid_search,
        questions
    )
]

for r in results:
    print(r)