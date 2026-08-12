from rag.chunking import chunk_document
from rag.embedding import EmbeddingModel
from rag.vector_store import VectorStore

embedder = EmbeddingModel()
store = VectorStore(dim=embedder.dim)


doc_text = open("Complete Enterprise Product Catalog.txt", encoding="utf-8").read()

chunks = chunk_document(
    doc_text,
    base_metadata={"doc": "Complete Enterprise Product Catalog"}
)

texts = [c["text"] for c in chunks]
vectors = embedder.embed_batch(texts)

for chunk, vector in zip(chunks, vectors):
    store.add(text=chunk["text"], embedding=vector, metadata=chunk["metadata"])


store.save("./data/marketloop_vector_db")
