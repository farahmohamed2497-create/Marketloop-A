from __future__ import annotations

import asyncio
import sqlite3

from aiohttp.test_utils import TestClient, TestServer

from platform_app.agents import AgentRouter
from platform_app.app import create_app
from platform_app.sessions import ChatSessionStore


async def _client(database) -> TestClient:
    store = ChatSessionStore(lambda: sqlite3.connect(database))
    router = AgentRouter(runners={
        "rag-agent": lambda message: f"rag:{message}",
        "state-graph-agent": lambda message: f"graph:{message}",
        "planning-agent": lambda message: f"plan:{message}",
    })
    server = TestServer(create_app(router=router, store=store))
    client = TestClient(server)
    await client.start_server()
    return client


def test_switching_agents_routes_to_the_selected_backend(tmp_path) -> None:
    async def exercise() -> None:
        client = await _client(tmp_path / "routing.db")
        try:
            agents = await (await client.get("/api/agents")).json()
            assert {agent["id"] for agent in agents["agents"]} == {"rag-agent", "state-graph-agent", "planning-agent"}

            graph = await client.post("/api/chat/messages", json={"agent_id": "state-graph-agent", "message": "open a case"})
            graph_body = await graph.json()
            planning = await client.post("/api/chat/messages", json={"agent_id": "planning-agent", "message": "audit sales"})
            planning_body = await planning.json()

            assert graph_body["reply"] == "graph:open a case"
            assert planning_body["reply"] == "plan:audit sales"
            assert graph_body["session_id"] != planning_body["session_id"]
        finally:
            await client.close()
    asyncio.run(exercise())


def test_history_persists_when_returning_to_an_agent(tmp_path) -> None:
    async def exercise() -> None:
        client = await _client(tmp_path / "history.db")
        try:
            first = await client.post("/api/chat/messages", json={"agent_id": "rag-agent", "message": "policy"})
            body = await first.json()
            history = await client.get(f"/api/chat/sessions/{body['session_id']}")
            assert [item["content"] for item in (await history.json())["history"]] == ["policy", "rag:policy"]
        finally:
            await client.close()
    asyncio.run(exercise())
