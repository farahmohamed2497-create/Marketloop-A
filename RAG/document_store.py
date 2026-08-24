"""Document-level admin operations over the RAG retrieval stack.

The retrieval code (naive_rag / hybrid_search / agentic_rag) works at the
level of individual chunks living in a VectorStore + KeywordStore. The
admin panel needs to work one level up: a document a human uploaded or
removed. This module is that layer.

Every mutation:
  1. updates the on-disk VectorStore (data/marketloop_vector_db/*) and the
     on-disk BM25 KeywordStore (data/keyword_store.pkl) so the change
     survives a process restart,
  2. records the document in the shared MarketLoop SQLite database
     (Rag_Documents) so the admin panel can list what's indexed, and
  3. bumps Rag_Store_Meta.version.

`rag_pipeline._ensure_fresh_stores()` compares its own cached version
against Rag_Store_Meta.version before every retrieval call and reloads
from disk when they differ. That's what makes an admin's add/remove
reach a running agent process's *next* query without restarting it,
including when the admin API and the agent are two different processes,
since the version lives in the database both of them already read.
"""

from __future__ import annotations

import pickle
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from mcp_server.db import get_connection
from mcp_server.tools.knowledge_store import KeywordStore

from rag.chunking import chunk_document
from rag.embedding import EmbeddingModel
from rag.vector_store import VectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTOR_DB_PATH = PROJECT_ROOT / "data" / "marketloop_vector_db"
KEYWORD_STORE_PATH = PROJECT_ROOT / "data" / "keyword_store.pkl"

DEFAULT_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2


class RagDocumentStoreError(RuntimeError):
    """Raised when a document-store operation can't be completed."""


class RagDocumentStore:
    """Admin-facing add/remove/list operations for the RAG knowledge base."""

    def __init__(
        self,
        *,
        connection_factory: Callable[[], Any] | None = None,
        embedder: EmbeddingModel | None = None,
    ) -> None:
        self._get_connection = connection_factory or get_connection
        self._embedder = embedder  # created lazily; tests can inject a fake
        self._ensure_tables()

    # ------------------------------------------------------------------
    # schema (self-ensuring, same pattern as tickets/service.py and
    # hitl/node.py elsewhere in this repo)
    # ------------------------------------------------------------------
    def _ensure_tables(self) -> None:
        with self._get_connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS Rag_Documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    added_at TEXT NOT NULL,
                    added_by TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS Rag_Store_Meta (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    version INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO Rag_Store_Meta (id, version) VALUES (1, 0)"
            )
            connection.commit()

    # ------------------------------------------------------------------
    # store loading helpers
    # ------------------------------------------------------------------
    def _get_embedder(self) -> EmbeddingModel:
        if self._embedder is None:
            self._embedder = EmbeddingModel()
        return self._embedder

    def _load_vector_store(self) -> VectorStore:
        if VECTOR_DB_PATH.exists():
            return VectorStore.load(str(VECTOR_DB_PATH))
        return VectorStore(dim=self._get_embedder().dim or DEFAULT_EMBEDDING_DIM)

    def _load_keyword_store(self) -> KeywordStore:
        if KEYWORD_STORE_PATH.exists():
            with KEYWORD_STORE_PATH.open("rb") as handle:
                return pickle.load(handle)
        return KeywordStore()

    def _save_keyword_store(self, store: KeywordStore) -> None:
        KEYWORD_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with KEYWORD_STORE_PATH.open("wb") as handle:
            pickle.dump(store, handle)

    def _bump_version(self, connection) -> None:
        connection.execute(
            "UPDATE Rag_Store_Meta SET version = version + 1 WHERE id = 1"
        )

    def get_version(self) -> int:
        with self._get_connection() as connection:
            row = connection.execute(
                "SELECT version FROM Rag_Store_Meta WHERE id = 1"
            ).fetchone()
        return int(row[0]) if row else 0

    # ------------------------------------------------------------------
    # admin operations
    # ------------------------------------------------------------------
    def list_documents(self) -> list[dict[str, Any]]:
        with self._get_connection() as connection:
            rows = connection.execute(
                """
                SELECT document_id, filename, chunk_count, added_at, added_by
                FROM Rag_Documents
                ORDER BY added_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def add_document(
        self,
        *,
        filename: str,
        text: str,
        added_by: str | None = None,
    ) -> dict[str, Any]:
        """Chunk, embed, and index a new document. Reflected on the very
        next retrieval call made by any process (see module docstring)."""

        if not text.strip():
            raise RagDocumentStoreError("Document text is empty.")

        document_id = str(uuid.uuid4())
        embedder = self._get_embedder()
        vector_store = self._load_vector_store()
        keyword_store = self._load_keyword_store()

        chunks = chunk_document(
            text,
            base_metadata={"document_id": document_id, "source": filename},
        )
        if not chunks:
            raise RagDocumentStoreError("Document produced no chunks.")

        embeddings = embedder.embed_batch([c["text"] for c in chunks])
        for chunk, embedding in zip(chunks, embeddings):
            vector_store.add(text=chunk["text"], embedding=embedding, metadata=chunk["metadata"])
            keyword_store.upsert(payload=chunk["text"], metadata=chunk["metadata"])

        vector_store.save(str(VECTOR_DB_PATH))
        self._save_keyword_store(keyword_store)

        added_at = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as connection:
            connection.execute(
                """
                INSERT INTO Rag_Documents (document_id, filename, chunk_count, added_at, added_by)
                VALUES (?, ?, ?, ?, ?)
                """,
                (document_id, filename, len(chunks), added_at, added_by),
            )
            self._bump_version(connection)
            connection.commit()

        return {
            "document_id": document_id,
            "filename": filename,
            "chunk_count": len(chunks),
            "added_at": added_at,
        }

    def remove_document(self, document_id: str) -> int:
        """Remove a document's chunks from both stores. Returns the number
        of chunks removed. Raises RagDocumentStoreError if unknown."""

        with self._get_connection() as connection:
            existing = connection.execute(
                "SELECT document_id FROM Rag_Documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()

        if existing is None:
            raise RagDocumentStoreError(f"Unknown RAG document: {document_id}")

        vector_store = self._load_vector_store()
        keyword_store = self._load_keyword_store()

        removed_vectors = vector_store.remove_by_metadata({"document_id": document_id})
        removed_keywords = keyword_store.remove_by_metadata({"document_id": document_id})

        vector_store.save(str(VECTOR_DB_PATH))
        self._save_keyword_store(keyword_store)

        with self._get_connection() as connection:
            connection.execute(
                "DELETE FROM Rag_Documents WHERE document_id = ?",
                (document_id,),
            )
            self._bump_version(connection)
            connection.commit()

        return max(len(removed_vectors), removed_keywords)