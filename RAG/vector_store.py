def data(self):
        return self._data
import uuid
import pickle
import os
import numpy as np
import hnswlib


class MetadataStore:
    

    def __init__(self):
        self._data = {}

    def add(self, item_id: str, metadata: dict, text: str):
        self._data[item_id] = {"metadata": metadata, "text": text}

    def get(self, item_id: str):
        return self._data.get(item_id)


class MetadataIndex:
    

    def __init__(self):
        self._index: dict[str, dict] = {}

    def add(self, item_id: str, metadata: dict):
        for field, value in metadata.items():
            self._index.setdefault(field, {}).setdefault(value, set()).add(item_id)

    def filter_ids(self, filters: dict) -> set:
        result = None
        for field, value in filters.items():
            matching = self._index.get(field, {}).get(value, set())
            result = matching if result is None else result & matching
        return result if result is not None else set()


class VectorStore:
    def __init__(self, dim: int, max_elements: int = 10000, space: str = "cosine"):
        self.dim = dim
        self.max_elements = max_elements
        self.space = space

        self.index = hnswlib.Index(space=space, dim=dim)
        self.index.init_index(max_elements=max_elements, ef_construction=200, M=16)
        self.index.set_ef(50)

        self.metadata_store = MetadataStore()
        self.metadata_index = MetadataIndex()

        self._next_int_id = 0
        self._id_map = {}          
        self._reverse_id_map = {} 

    def add(self, text: str, embedding: np.ndarray, metadata: dict, item_id: str = None) -> str:
        item_id = item_id or str(uuid.uuid4())
        int_id = self._next_int_id
        self._next_int_id += 1

        self.index.add_items(np.array([embedding]), np.array([int_id]))
        self._id_map[int_id] = item_id
        self._reverse_id_map[item_id] = int_id

        self.metadata_store.add(item_id, metadata, text)
        self.metadata_index.add(item_id, metadata)

        return item_id

    def search(self, query_embedding: np.ndarray, k: int = 5, filters: dict | None = None):
        
        allowed_ids = None
        if filters:
            allowed_ids = self.metadata_index.filter_ids(filters)
            if not allowed_ids:
                return []

        def hnsw_filter(int_id):
            if allowed_ids is None:
                return True
            return self._id_map.get(int_id) in allowed_ids

        labels, distances = self.index.knn_query(
            np.array([query_embedding]), k=k, filter=hnsw_filter
        )

        results = []
        for int_id, dist in zip(labels[0], distances[0]):
            str_id = self._id_map[int_id]
            payload = self.metadata_store.get(str_id)
            results.append({
                "id": str_id,
                "score": 1 - dist,
                "text": payload["text"],
                "metadata": payload["metadata"],
            })
        return results

    # ---------- Persistenc----------

    def save(self, path: str):
        
        os.makedirs(path, exist_ok=True)

        self.index.save_index(os.path.join(path, "hnsw_index.bin"))

        with open(os.path.join(path, "metadata_store.pkl"), "wb") as f:
            pickle.dump(self.metadata_store._data, f)

        with open(os.path.join(path, "id_maps.pkl"), "wb") as f:
            pickle.dump({
                "id_map": self._id_map,
                "reverse_id_map": self._reverse_id_map,
                "next_int_id": self._next_int_id,
            }, f)

        with open(os.path.join(path, "config.pkl"), "wb") as f:
            pickle.dump({
                "dim": self.dim,
                "max_elements": self.max_elements,
                "space": self.space,
            }, f)

    @classmethod
    def load(cls, path: str) -> "VectorStore":
       
        with open(os.path.join(path, "config.pkl"), "rb") as f:
            config = pickle.load(f)

        store = cls(
            dim=config["dim"],
            max_elements=config["max_elements"],
            space=config["space"],
        )

        store.index.load_index(
            os.path.join(path, "hnsw_index.bin"),
            max_elements=config["max_elements"],
        )

        with open(os.path.join(path, "metadata_store.pkl"), "rb") as f:
            store.metadata_store._data = pickle.load(f)

        with open(os.path.join(path, "id_maps.pkl"), "rb") as f:
            saved = pickle.load(f)
            store._id_map = saved["id_map"]
            store._reverse_id_map = saved["reverse_id_map"]
            store._next_int_id = saved["next_int_id"]

        for item_id, payload in store.metadata_store._data.items():
            store.metadata_index.add(item_id, payload["metadata"])

        return store
    