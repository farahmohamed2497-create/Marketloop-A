"""
Episodic memory store.

The promote-or-drop router writes PROMOTE decisions here.
A separate consolidation layer is responsible for reading these
episodes and building semantic memory.

This module intentionally contains no semantic-memory logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable


@dataclass(slots=True)
class EpisodicMemory:
    content: str
    role: str
    reasoning: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class EpisodicStore:
    """Simple in-memory episodic memory store."""

    def __init__(self) -> None:
        self._episodes: list[EpisodicMemory] = []

    def add(self, episode: EpisodicMemory) -> None:
        """Store one promoted memory."""
        self._episodes.append(episode)

    def extend(self, episodes: Iterable[EpisodicMemory]) -> None:
        """Store multiple promoted memories."""
        self._episodes.extend(episodes)

    def get_all(self) -> list[EpisodicMemory]:
        """Return a copy of all stored episodes."""
        return list(self._episodes)

    def count(self) -> int:
        """Return the number of stored episodes."""
        return len(self._episodes)

    def clear(self) -> None:
        """Remove all stored episodes."""
        self._episodes.clear()

    def __len__(self) -> int:
        """Support len(store)."""
        return len(self._episodes)

    def __iter__(self):
        """Support iteration over the store."""
        return iter(self._episodes)

    def __getitem__(self, index: int) -> EpisodicMemory:
        """Support indexing, e.g. store[0]."""
        return self._episodes[index]    