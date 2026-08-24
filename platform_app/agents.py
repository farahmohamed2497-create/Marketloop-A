from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


AgentRunner = Callable[[str], str]


@dataclass(frozen=True)
class AgentDefinition:
    agent_id: str
    name: str
    description: str


class AgentRouter:
    """Route a platform message to an explicitly registered project agent."""

    def __init__(self, runners: dict[str, AgentRunner] | None = None) -> None:
        self._definitions = {
            "memory-rag": AgentDefinition("memory-rag", "Memory & RAG", "Grounded knowledge and memory agent"),
            "state-graph": AgentDefinition("state-graph", "State Graph", "Persistent workflow agent"),
            "planning": AgentDefinition("planning", "Planning", "Sales-audit planning agent"),
        }
        self._runners = runners or {"memory-rag": self._run_rag}

    def list_agents(self) -> list[dict[str, Any]]:
        return [
            {"id": item.agent_id, "name": item.name, "description": item.description,
             "available": item.agent_id in self._runners}
            for item in self._definitions.values()
        ]

    def route(self, agent_id: str, message: str) -> str:
        if agent_id not in self._definitions:
            raise ValueError("Unknown agent.")
        runner = self._runners.get(agent_id)
        if runner is None:
            raise RuntimeError(f"{self._definitions[agent_id].name} is not yet exposed as a chat runner.")
        return runner(message)

    @staticmethod
    def _run_rag(message: str) -> str:
        from rag.rag_pipeline import answer_with_hybrid

        return str(answer_with_hybrid(message)["answer"])
