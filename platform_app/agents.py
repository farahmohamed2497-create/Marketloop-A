from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import os
from typing import Any

from mcp_server import agent_registry


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
            "rag-agent": AgentDefinition("rag-agent", "Memory & RAG", "Grounded knowledge and memory agent"),
            "state-graph-agent": AgentDefinition("state-graph-agent", "State Graph", "Persistent workflow agent"),
            "planning-agent": AgentDefinition("planning-agent", "Planning", "Sales-audit planning agent"),
        }
        self._uses_live_registry = runners is None
        self._runners = runners or {
            "rag-agent": self._run_rag,
            "state-graph-agent": self._run_state_graph,
            "planning-agent": self._run_planning,
        }
        if self._uses_live_registry:
            for definition in self._definitions.values():
                agent_registry.ensure_agent(definition.agent_id, definition.name, definition.agent_id)

    def list_agents(self) -> list[dict[str, Any]]:
        live = (
            {agent["agent_id"] for agent in agent_registry.list_agents()}
            if self._uses_live_registry else set(self._runners)
        )
        return [{"id": item.agent_id, "name": item.name, "description": item.description,
                 "available": item.agent_id in self._runners, "live": item.agent_id in live}
                for item in self._definitions.values()]

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

    @staticmethod
    def _chat_model():
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is required to run this agent.")
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    def _run_state_graph(self, message: str) -> str:
        """Run the real Graph 1 workflow; its state is persisted by its engine."""
        from uuid import uuid4
        from state_graph.factory import create_graph1
        from state_graph.graph1.graph import create_initial_state

        engine = create_graph1(llm=self._chat_model())
        final = engine.run(create_initial_state(str(uuid4()), message), max_steps=20)
        return json.dumps({"run_id": final.run_id, "status": final.status,
                           "current_node": final.current_node, "outputs": final.outputs})

    def _run_planning(self, message: str) -> str:
        """Route the request through the project planning agent, not a mock."""
        from agent.planning_agent import SalesAuditPlanningAgent
        return SalesAuditPlanningAgent(self._chat_model()).solve(message)
