# Platform F3 — Agent Switching and Chat

This document covers **only F3**. Tool management (F1) and RAG-document,
HITL, and ticket administration (F2) are deliberately outside this slice.

## API contract

- `GET /api/agents` returns the registered, live agent adapters.
- `POST /api/chat/messages` accepts `agent_id`, `message`, and optional
  `session_id`; it returns the persisted user/assistant messages.
- `GET /api/chat/sessions/{session_id}` returns the conversation history.

Every agent adapter receives the same `message` string and owns its actual
execution. `rag-agent` runs the existing Hybrid RAG pipeline,
`state-graph-agent` runs Graph 1, and `planning-agent` runs the existing sales
audit planning agent. The platform never invents an answer: missing provider
credentials are returned as a clear error from that adapter.

## Session model

F3 owns `Platform_Chat_Sessions` and `Platform_Chat_Messages`. A session is
scoped to one selected agent, so switching agents creates or resumes that
agent's own thread rather than mixing unrelated histories.

## Deferred dependencies

F3 does not implement F1/F2 APIs. HITL/ticket visibility stays owned by F2;
F3 only returns the Graph 1 run ID/status so an admin can resolve a pause from
the existing admin surface.
