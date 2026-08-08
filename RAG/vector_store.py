"""HNSW vector store with metadata payload and pre-search filtering."""

from __future__ import annotations

import os
import pickle
import uuid
from typing import Any

import hnswlib
import numpy as np


class MetadataStore:
    """Stores the text payload and metadata associated with each vector."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    def add(self, item_id: str, metadata: dict[str, Any], text: str) -> None:
        self._data[item_id] = {"metadata": metadata, "text": text}

    def get(self, item_id: str) -> dict[str, Any] | None:
        return self._data.get(item_id)

    @property
    def data(self) -> dict[str, dict[str, Any]]:
        return self._data


class MetadataIndex:
    """Inverted metadata index used to filter candidates inside HNSW search."""

    def __init__(self) -> None:
        self._index: dict[str, dict[Any, set[str]]] = {}

    def add(self, item_id: str, metadata: dict[str, Any]) -> None:
        for field, value in metadata.items():
            self._index.setdefault(field, {}).setdefault(value, set()).add(item_id)

    def filter_ids(self, filters: dict[str, Any] | None) -> set[str] | None:
        if not filters:
            return None

        matches: set[str] | None = None
        for field, value in filters.items():
            field_matches = self._index.get(field, {}).get(value, set())
            matches = set(field_matches) if matches is None else matches & field_matches
        return matches or set()


class VectorStore:
    """Local cosine-similarity HNSW index with payload and metadata indexes."""

    def __init__(self, dim: int, max_elements: int = 10_000, space: str = "cosine") -> None:
        self.dim = dim
        self.max_elements = max_elements
        self.space = space

        self.index = hnswlib.Index(space=space, dim=dim)
        self.index.init_index(max_elements=max_elements, ef_construction=200, M=16)
        self.index.set_ef(50)

        self.metadata_store = MetadataStore()
        self.metadata_index = MetadataIndex()
        self._next_int_id = 0
        self._id_map: dict[int, str] = {}
        self._reverse_id_map: dict[str, int] = {}

    @property
    def size(self) -> int:
        return len(self._id_map)

    def add(
        self,
        text: str,
        embedding: np.ndarray,
        metadata: dict[str, Any],
        item_id: str | None = None,
    ) -> str:
        if self.size >= self.max_elements:
            raise ValueError("VectorStore capacity exceeded.")

        item_id = item_id or str(uuid.uuid4())
        int_id = self._next_int_id
        self._next_int_id += 1

        self.index.add_items(np.asarray([embedding]), np.asarray([int_id]))
        self._id_map[int_id] = item_id
        self._reverse_id_map[item_id] = int_id
        self.metadata_store.add(item_id, metadata, text)
        self.metadata_index.add(item_id, metadata)
        return item_id

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search ANN candidates, applying metadata filters during search."""
        allowed_ids = self.metadata_index.filter_ids(filters)
        available_count = len(allowed_ids) if allowed_ids is not None else self.size
        if available_count == 0:
            return []

        effective_k = min(k, available_count)

        def hnsw_filter(int_id: int) -> bool:
            return allowed_ids is None or self._id_map.get(int_id) in allowed_ids

        labels, distances = self.index.knn_query(
            np.asarray([query_embedding]),
            k=effective_k,
            filter=hnsw_filter,
        )

        results: list[dict[str, Any]] = []
        for int_id, distance in zip(labels[0], distances[0]):
            item_id = self._id_map[int(int_id)]
            payload = self.metadata_store.get(item_id)
            if payload is None:
                continue
            results.append(
                {
                    "id": item_id,
                    "score": float(1 - distance),
                    "text": payload["text"],
                    "metadata": payload["metadata"],
                }
            )
        return results

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.index.save_index(os.path.join(path, "hnsw_index.bin"))

        with open(os.path.join(path, "metadata_store.pkl"), "wb") as handle:
            pickle.dump(self.metadata_store.data, handle)
        with open(os.path.join(path, "id_maps.pkl"), "wb") as handle:
            pickle.dump(
                {
                    "id_map": self._id_map,
                    "reverse_id_map": self._reverse_id_map,
                    "next_int_id": self._next_int_id,
                },
                handle,
            )
        with open(os.path.join(path, "config.pkl"), "wb") as handle:
            pickle.dump(
                {"dim": self.dim, "max_elements": self.max_elements, "space": self.space},
                handle,
            )

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        with open(os.path.join(path, "config.pkl"), "rb") as handle:
            config = pickle.load(handle)

        store = cls(
            dim=config["dim"],
            max_elements=config["max_elements"],
            space=config["space"],
        )
        store.index.load_index(
            os.path.join(path, "hnsw_index.bin"),
            max_elements=config["max_elements"],
        )

        with open(os.path.join(path, "metadata_store.pkl"), "rb") as handle:
            store.metadata_store._data = pickle.load(handle)
        with open(os.path.join(path, "id_maps.pkl"), "rb") as handle:
            saved = pickle.load(handle)
            store._id_map = saved["id_map"]
            store._reverse_id_map = saved["reverse_id_map"]
            store._next_int_id = saved["next_int_id"]

        for item_id, payload in store.metadata_store.data.items():
            store.metadata_index.add(item_id, payload["metadata"])
        return store