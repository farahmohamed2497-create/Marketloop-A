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

from rag.document_store import RagDocumentStore, RagDocumentStoreError
from state_graph.hitl.node import HITLNode
from state_graph.tickets.service import FailureTicketService
from state_graph.engine_registry import resume_run, UnknownGraphError

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


class AddRagDocumentRequest(BaseModel):
    filename: str
    text: str


class ResolveHitlRequest(BaseModel):
    decision: str  # "approve" | "reject"


class UpdateTicketRequest(BaseModel):
    status: str  # "investigating" | "resolved"
    resolution: str | None = None


rag_documents: RagDocumentStore | None = None
hitl: HITLNode | None = None
failure_tickets: FailureTicketService | None = None


def get_rag_documents() -> RagDocumentStore:
    """Lazily construct the RAG document store.

    Deferred (rather than built at import time) so tests can monkeypatch
    mcp_server.db.get_database_path *before* the first table-creation
    write happens, the same way rag_pipeline.py defers its clients so it
    "can be imported by tests without" touching real external state.
    """
    global rag_documents
    if rag_documents is None:
        rag_documents = RagDocumentStore()
    return rag_documents


def get_hitl() -> HITLNode:
    global hitl
    if hitl is None:
        hitl = HITLNode()
    return hitl


def get_failure_tickets() -> FailureTicketService:
    global failure_tickets
    if failure_tickets is None:
        failure_tickets = FailureTicketService()
    return failure_tickets


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


# ---------------------------------------------------------------------
# RAG document management
# ---------------------------------------------------------------------
# These write straight to the same VectorStore + KeywordStore files the
# Memory & RAG agent reads from (data/marketloop_vector_db, data/
# keyword_store.pkl) and bump Rag_Store_Meta.version in the shared DB.
# rag_pipeline._ensure_fresh_stores() checks that version before every
# retrieval call, so a running agent picks up the change on its next
# query without a restart. See RAG/document_store.py for the full
# explanation.


@app.get("/api/rag/documents")
def list_rag_documents():
    """Every document currently indexed in the RAG knowledge base."""
    return get_rag_documents().list_documents()


@app.post("/api/rag/documents", status_code=201)
def add_rag_document(body: AddRagDocumentRequest):
    """Chunk, embed, and index a new document."""
    try:
        return get_rag_documents().add_document(filename=body.filename, text=body.text, added_by="admin")
    except RagDocumentStoreError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/rag/documents/{document_id}")
def remove_rag_document(document_id: str):
    """Remove a document's chunks from both the vector store and the
    keyword store, and drop it from Rag_Documents."""
    try:
        removed_chunks = get_rag_documents().remove_document(document_id)
    except RagDocumentStoreError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {"document_id": document_id, "removed_chunks": removed_chunks}


# ---------------------------------------------------------------------
# Human-in-the-loop tasks
# ---------------------------------------------------------------------
# A HITL request is an EXPECTED pause for a decision a graph is not
# allowed to make on its own (see state_graph/hitl/policy.py for the
# thresholds). Resolving one records the admin's decision in
# HITL_Requests and then resumes the run — the resumed run reads that
# decision back out (see e.g. ShippingGraph._resume_from_hitl_decision),
# it does not just proceed as if nothing happened.


@app.get("/api/hitl/tasks")
def list_hitl_tasks(status: str | None = None):
    """Pending (or, with ?status=resolved, past) HITL requests, across
    every graph, for the admin inbox."""
    try:
        return get_hitl().list_requests(status=status)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/hitl/tasks/{request_id}")
def get_hitl_task(request_id: str):
    """One HITL request plus the graph's persisted state at the moment
    it paused — what the admin needs to actually make the call."""
    request = get_hitl().get_request(request_id)
    if request is None:
        raise HTTPException(status_code=404, detail=f"Unknown HITL request: {request_id}")
    return request


@app.post("/api/hitl/tasks/{request_id}/resolve")
def resolve_hitl_task(request_id: str, body: ResolveHitlRequest):
    """Record the admin's decision and resume the paused run.

    This is the only path that unpauses a HITL'd run — there is no
    auto-approve anywhere in the graph code.
    """
    request = get_hitl().get_request(request_id)
    if request is None:
        raise HTTPException(status_code=404, detail=f"Unknown HITL request: {request_id}")

    try:
        get_hitl().resolve(request_id, body.decision)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        resumed = resume_run(request["run_id"], request["graph_name"])
    except UnknownGraphError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "request_id": request_id,
        "decision": body.decision,
        "run_id": resumed.run_id,
        "status": resumed.status,
        "current_node": resumed.current_node,
        "outputs": resumed.outputs,
    }


# ---------------------------------------------------------------------
# Failure tickets
# ---------------------------------------------------------------------
# A ticket is an UNPLANNED failure a single retry can't fix — a tool
# call errored, a schema validation failed, the model returned
# something the graph couldn't act on (see StateGraphEngine.step()'s
# except branch, which is the only code path that creates one). This is
# a distinct lifecycle from a HITL pause: open -> investigating ->
# resolved, and only "resolved" is allowed to resume the run, and only
# from the exact node it failed on.


@app.get("/api/tickets")
def list_tickets(status: str | None = None):
    """Failure tickets, optionally filtered by status, for the ticket board."""
    try:
        return get_failure_tickets().list_tickets(status=status)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/tickets/{ticket_id}")
def get_ticket(ticket_id: str):
    """One ticket plus the graph's persisted state at the moment it failed."""
    ticket = get_failure_tickets().get_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Unknown ticket: {ticket_id}")
    return ticket


@app.patch("/api/tickets/{ticket_id}")
def update_ticket(ticket_id: str, body: UpdateTicketRequest):
    """Move a ticket through open -> investigating -> resolved.

    Resolving a ticket (status="resolved") also resumes the run from
    its last checkpoint — the exact node it failed on, not a restart
    from the top.
    """
    ticket = get_failure_tickets().get_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Unknown ticket: {ticket_id}")

    try:
        get_failure_tickets().set_ticket_status(ticket_id, body.status, resolution=body.resolution)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    response: dict = {"ticket_id": ticket_id, "status": body.status}

    if body.status == "resolved":
        try:
            resumed = resume_run(ticket["run_id"], ticket["graph_name"])
        except UnknownGraphError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        response["resumed_run"] = {
            "run_id": resumed.run_id,
            "status": resumed.status,
            "current_node": resumed.current_node,
            "outputs": resumed.outputs,
        }

    return response
