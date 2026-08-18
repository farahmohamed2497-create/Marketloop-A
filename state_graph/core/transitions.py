from __future__ import annotations

from collections.abc import Mapping

from .exceptions import InvalidTransitionError


class TransitionTable:
    """
    Explicit transition table for a state graph.
    """

    def __init__(self) -> None:
        self._transitions: dict[
            str,
            set[str],
        ] = {}

    def add(
        self,
        source: str,
        target: str,
    ) -> None:
        self._transitions.setdefault(
            source,
            set(),
        ).add(target)

    def add_many(
        self,
        source: str,
        targets: list[str],
    ) -> None:
        for target in targets:
            self.add(source, target)

    def allowed(
        self,
        source: str,
        target: str,
    ) -> bool:
        return target in self._transitions.get(
            source,
            set(),
        )

    def validate(
        self,
        source: str,
        target: str,
    ) -> None:
        if not self.allowed(source, target):
            raise InvalidTransitionError(
                f"Invalid transition: "
                f"{source!r} -> {target!r}"
            )

    def as_mapping(
        self,
    ) -> Mapping[str, set[str]]:
        return dict(self._transitions)