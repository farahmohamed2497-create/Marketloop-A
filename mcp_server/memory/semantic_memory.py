"""
Semantic memory store.
Facts here are written ONLY by the ConsolidationLayer, never directly
by the promote-or-drop router or the live agent loop.
Each fact is versioned: updates never overwrite history, they supersede it.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict


@dataclass(slots=True)
class SemanticFact:
    id: str
    key: str
    value: str
    version: int
    created_at: float
    valid_until: float | None = None
    superseded_by: str | None = None
    source_episodes: list[str] = field(default_factory=list)


class SemanticMemory:
    """In-memory semantic fact store, grouped by key."""

    def __init__(self) -> None:
        self._facts: dict[str, list[SemanticFact]] = {}
        self._next_id = 1
 
    def _gen_id(self) -> str:
        fact_id = f"fact_{self._next_id}"
        self._next_id += 1
        return fact_id

    def get_current(self, key: str) -> SemanticFact | None:
        versions = self._facts.get(key, [])
        now = time.time()

        for fact in reversed(versions):
            if fact.superseded_by is not None:
                continue
            if fact.valid_until is not None and fact.valid_until < now:
                continue
            return fact

        return None

    def get_history(self, key: str) -> list[SemanticFact]:
        return list(self._facts.get(key, []))

    def write_fact(
        self,
        key: str,
        value: str,
        source_episodes: list[str],
        valid_until: float | None = None,
    ) -> SemanticFact:
        existing = self.get_current(key)

        new_fact = SemanticFact(
            id=self._gen_id(),
            key=key,
            value=value,
            version=(existing.version + 1) if existing else 1,
            created_at=time.time(),
            valid_until=valid_until,
            source_episodes=source_episodes,
        )

        if existing:
            existing.superseded_by = new_fact.id

        self._facts.setdefault(key, []).append(new_fact)
        return new_fact

    def expire_stale_facts(self) -> list[SemanticFact]:
        now = time.time()
        expired = []

        for versions in self._facts.values():
            for fact in versions:
                if (
                    fact.valid_until is not None
                    and fact.valid_until < now
                    and fact.superseded_by is None
                ):
                    expired.append(fact)

        return expired

    def get_all_current(self) -> list[SemanticFact]:
        result = []
        for key in self._facts:
            current = self.get_current(key)
            if current:
                result.append(current)
        return result


    def save(self, path: str) -> None:

        data = {
            "next_id": self._next_id,
            "facts": {
                key: [asdict(fact) for fact in versions]
                for key, versions in self._facts.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "SemanticMemory":

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        memory = cls()
        memory._next_id = data["next_id"]
        memory._facts = {
            key: [SemanticFact(**fact_dict) for fact_dict in versions]
            for key, versions in data["facts"].items()
        }
        return memory