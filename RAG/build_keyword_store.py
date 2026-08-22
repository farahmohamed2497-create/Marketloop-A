from pathlib import Path
from RAG.chunking import chunk_document
from mcp_server.tools.knowledge_store import KeywordStore
import pickle
import os

store = KeywordStore()

BASE_DIR = Path(__file__).resolve().parent.parent

catalog_path = BASE_DIR / "Complete Enterprise Product Catalog.txt"

with open(
    catalog_path,
    encoding="utf-8"
) as f:
    doc_text = f.read()

chunks = chunk_document(
    doc_text,
    base_metadata={
        "doc": "Complete Enterprise Product Catalog"
    }
)

for chunk in chunks:
    store.upsert(
        payload=chunk["text"],
        metadata=chunk["metadata"]
    )

os.makedirs(BASE_DIR / "data", exist_ok=True)

with open(
    BASE_DIR / "data" / "keyword_store.pkl",
    "wb"
) as f:
    pickle.dump(store, f)

print(
    f"Indexed {len(chunks)} chunks into BM25 store."
)
