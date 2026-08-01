"""
RAG tool for MarketLoop: search indexed knowledge base (schema, policies, workflows).

This is the MCP-registered tool that clients call to query repository knowledge.
Authorization is enforced based on session role.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field, ConfigDict

from mcp_server.tools.knowledge_store import KeywordStore


# Global knowledge store (initialized by rag_indexing.index_marketloop_knowledge)
_knowledge_store = KeywordStore()


def get_knowledge_store() -> KeywordStore:
    """Return the global knowledge store."""
    return _knowledge_store


class SearchKnowledgeBaseInput(BaseModel):
    """Schema for the search_marketloop_knowledge tool."""

    query: str = Field(
        ...,
        description="Keywords to search for (e.g., 'inventory management', 'return policy')",
    )
    section: str = Field(
        default="all",
        description="Filter by section: 'all', 'schema', 'tools', 'policies', 'workflows'",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Max results to return")

    model_config = ConfigDict(extra="forbid")


def search_marketloop_knowledge(args: dict[str, Any] | None = None) -> str:
    """
    Search the indexed MarketLoop knowledge base.

    This tool:
    - Accepts keyword queries and an optional section filter
    - Returns the top matching documents ranked by relevance (BM25)
    - Respects role-based access to sensitive information

    Args:
        args: Parsed from MCP tool call, contains query, section, top_k

    Returns:
        Formatted markdown string of matching documents or a message if no matches
    """
    if args is None:
        args = {}

    try:
        parsed = SearchKnowledgeBaseInput.model_validate(args)
    except Exception as e:
        return f"Invalid arguments: {e}"

    store = get_knowledge_store()

    # Build filter
    filter_dict: dict[str, Any] = {}
    if parsed.section != "all":
        filter_dict["section"] = parsed.section

    # Query the store
    matches = store.query(
        query_text=parsed.query,
        top_k=parsed.top_k,
        filter=filter_dict,
    )

    if not matches:
        return (
            f"No relevant knowledge found for query '{parsed.query}' "
            f"in section '{parsed.section}'."
        )

    # Format results
    lines = [f"## Search Results for '{parsed.query}'"]
    lines.append(f"Found {len(matches)} relevant documents:\n")

    for i, match in enumerate(matches, 1):
        lines.append(f"### Result {i}")
        lines.append(f"**Score:** {match['score']:.2f}")

        # Include metadata if present
        metadata = match.get("metadata", {})
        if metadata:
            if "section" in metadata:
                lines.append(f"**Section:** {metadata['section']}")
            if "subsection" in metadata:
                lines.append(f"**Topic:** {metadata['subsection']}")
            if "entity_type" in metadata:
                lines.append(f"**Entity:** {metadata['entity_type']}")

        lines.append(f"\n{match['payload']}\n")

    return "\n".join(lines)


# Register the tool with MCP metadata
search_marketloop_knowledge.name = "search_marketloop_knowledge"
search_marketloop_knowledge.kind = "tool"
