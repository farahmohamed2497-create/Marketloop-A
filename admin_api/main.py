"""F1 admin API: agent + tool management.

Run from the repo root (so the `mcp_server` package import resolves):

    uvicorn admin_api.main:app --reload --port 8001

This talks directly to the same SQLite database the MCP server
processes read from. It does NOT hold an MCP client connection itself
-- it just writes to Agent_Tools, and each running agent process picks
up the change on its next list_tools/call_tool (see
_refresh_visible_tools in mcp_server/server.py).
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from mcp_server import agent_registry
from mcp_server.tool_catalog import discover_all_tool_names

app = FastAPI(title="MarketLoop Admin API")

# Plain HTML/JS admin UI will likely be served from a different
# origin/port during development. Lock this down to the real admin
# host before shipping.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateAgentRequest(BaseModel):
    agent_id: str
    display_name: str | None = None
    agent_type: str | None = None


class SetToolEnabledRequest(BaseModel):
    enabled: bool


@app.get("/api/agents")
def list_agents():
    """Every agent the admin panel knows about."""
    return agent_registry.list_agents()


@app.post("/api/agents", status_code=201)
def create_agent(body: CreateAgentRequest):
    """Register a new agent (e.g. before its process first connects)."""
    agent_registry.ensure_agent(
        body.agent_id,
        display_name=body.display_name,
        agent_type=body.agent_type,
    )
    return {"agent_id": body.agent_id, "status": "created"}


@app.get("/api/agents/{agent_id}/tools")
def get_agent_tools(agent_id: str):
    """Every known tool, with this agent's enabled/disabled state."""
    agents = {a["agent_id"] for a in agent_registry.list_agents()}
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail=f"Unknown agent: {agent_id}")

    all_tool_names = discover_all_tool_names()
    return agent_registry.get_agent_tool_state(agent_id, all_tool_names)


@app.put("/api/agents/{agent_id}/tools/{tool_name}")
def set_tool_enabled(agent_id: str, tool_name: str, body: SetToolEnabledRequest):
    """Enable or disable one tool for one agent. Reaches the live MCP
    server process on its next request — no redeploy."""
    all_tool_names = discover_all_tool_names()
    if tool_name not in all_tool_names:
        raise HTTPException(status_code=404, detail=f"Unknown tool: {tool_name}")

    try:
        agent_registry.set_tool_enabled(
            agent_id,
            tool_name,
            enabled=body.enabled,
            updated_by="admin",  # swap for the real admin user id once auth exists
        )
    except agent_registry.AgentRegistryError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "agent_id": agent_id,
        "tool_name": tool_name,
        "enabled": body.enabled,
    }


@app.get("/api/tools")
def list_all_tools():
    """Full tool catalog, for populating an "add tool" dropdown in the UI."""
    return discover_all_tool_names()