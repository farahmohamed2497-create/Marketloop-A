"""Build the local vector index manually; this is not a pytest module."""

from pathlib import Path


def build_vector_database() -> None:
    """Embed the catalog and write the local development index."""

    from RAG.chunking import chunk_document
    from RAG.embedding import EmbeddingModel
    from RAG.vector_store import VectorStore

    project_root = Path(__file__).resolve().parents[1]

    embedder = EmbeddingModel()
    store = VectorStore(dim=embedder.dim)

    doc_text = (project_root / "Complete Enterprise Product Catalog.txt").read_text(
        encoding="utf-8"
    )

    chunks = chunk_document(
        doc_text,
        base_metadata={"doc": "Complete Enterprise Product Catalog"},
    )

    vectors = embedder.embed_batch([chunk["text"] for chunk in chunks])

    for chunk, vector in zip(chunks, vectors):
        store.add(
            text=chunk["text"],
            embedding=vector,
            metadata=chunk["metadata"],
        )

    store.save(str(project_root / "data" / "marketloop_vector_db"))


if __name__ == "__main__":
    build_vector_database()
