"""Per-process registry of live StateGraphEngine instances, one per
graph_name ("refund", "shipping", "dispute").

The admin platform doesn't run graphs interactively — it resolves a HITL
request or a failure ticket and then needs to push that decision back
into the *same* run so it can continue past the node it paused/failed
on. `StateGraphEngine.resume(run_id)` already does the state-machine
part of that (see core/engine.py); this module just supplies the one
missing piece: turning a `graph_name` string on a persisted checkpoint
into the engine that knows how to execute that graph's nodes.

Engines are stateless with respect to any particular run — all run
state lives in State_Checkpoints — so one process-lifetime engine per
graph is safe to reuse across many resume() calls.
"""

from __future__ import annotations

from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel

from state_graph.core.engine import StateGraphEngine
from state_graph.core.models import GraphState
from state_graph.factory import create_graph1, create_graph2, create_graph3

_engines: dict[str, StateGraphEngine] = {}
_llm: BaseChatModel | None = None


class UnknownGraphError(ValueError):
    """Raised when a checkpoint references a graph_name we don't build."""


def _get_llm() -> BaseChatModel:
    """Lazily create the shared LLM client, matching the pattern already
    used elsewhere in this repo (rag_pipeline._get_groq_client(),
    planning_lab's ChatGroq/ChatMistralAI construction) — created only
    when a real resume actually needs it, never at import time."""
    global _llm
    if _llm is None:
        from langchain_groq import ChatGroq

        _llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    return _llm


_BUILDERS: dict[str, Callable[[BaseChatModel], StateGraphEngine]] = {
    "refund": lambda llm: create_graph1(llm=llm),
    "shipping": lambda llm: create_graph2(llm=llm),
    "dispute": lambda llm: create_graph3(llm),
}


def get_engine(graph_name: str) -> StateGraphEngine:
    """Return the (lazily-built, cached) engine for this graph_name."""
    if graph_name not in _BUILDERS:
        raise UnknownGraphError(
            f"No engine builder registered for graph_name={graph_name!r}. "
            f"Known graphs: {sorted(_BUILDERS)}"
        )

    if graph_name not in _engines:
        _engines[graph_name] = _BUILDERS[graph_name](_get_llm())

    return _engines[graph_name]


def resume_run(run_id: str, graph_name: str) -> GraphState:
    """Resume a waiting (HITL) or failed (ticket) run from its latest
    checkpoint, after the admin has recorded a decision / resolution.

    This is the function the admin API calls right after
    HITLNode.resolve(...) or FailureTicketService.resolve_ticket(...).
    """
    engine = get_engine(graph_name)
    return engine.resume(run_id)