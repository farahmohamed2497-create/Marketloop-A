# Platform F3 — Agent Switching and Chat

This document covers **only F3**. Tool management (F1) and RAG-document,
HITL, and ticket administration (F2) are deliberately outside this slice.

## API contract

- `GET /api/agents` returns the registered, live agent adapters.
- `POST /api/chat/messages` accepts `agent_id`, `message`, and optional
  `session_id`; it returns the persisted user/assistant messages.
- `GET /api/chat/sessions/{session_id}` returns the conversation history.

Every agent adapter receives the same `message` string and owns its actual
execution. The platform never invents an agent answer: missing credentials or
an unavailable integration are returned as a clear error from that adapter.

## Session model

F3 owns `Platform_Chat_Sessions` and `Platform_Chat_Messages`. A session is
scoped to one selected agent, so switching agents creates or resumes that
agent's own thread rather than mixing unrelated histories.

## Deferred dependencies

F3 does not implement F1/F2 APIs. The state-graph adapter is exposed only
when a graph-chat runner is registered; its eventual HITL/ticket visibility is
the responsibility of F2. RAG and planning adapters call their existing
project services and report missing provider credentials instead of returning
mock data.
