"""
MarketLoop MCP client (agent).

Connects to mcp_server/server.py over stdio (local dev) or Streamable HTTP
(remote deployment) and demonstrates every protocol concern required by the
assignment:

  - Capability negotiation : checks the server's declared capabilities from
                              initialize() before relying on them.
  - Notifications          : reacts live to tools/list_changed pushed by the
                              server when a user's role changes.
  - Elicitation            : answers elicitation/create requests from the
                              server (e.g. sign-off on a return/refund).
  - Sampling               : answers sampling/createMessage requests from the
                              server (e.g. generating a delay apology email).
  - Resources              : lists + reads resources (policy docs, reports).
  - Prompts                : lists + fetches parameterized prompt templates.
  - Progress tracking      : shows live progress for long-running tools.
  - Transport (Both)       : stdio for development, Streamable HTTP for
                              remote deployment.
  - RAG (hybrid + agentic) : answers knowledge questions grounded in the
                              vector/keyword stores, verified by Self-RAG
                              before being shown to the user.

Run with:
    python agent/client.py                      # stdio (default)
    python agent/client.py --transport http     # Streamable HTTP
    python agent/client.py --server-url http://localhost:8000/mcp
(from the repo root, with the mcp_server package importable, i.e. run
 `pip install -e .` first or run from a directory where `mcp_server` is on
 the Python path)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys 
import os
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.context import ClientRequestContext
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import (
    CreateMessageRequestParams,
    CreateMessageResult,
    ElicitRequestParams,
    ElicitResult,
    ProgressNotification,
    TextContent,
)
from mcp_server.memory.rolling_buffer import RollingBuffer
from mcp_server.memory.scratchpad import Scratchpad
from mcp_server.memory.promote_drop_router import PromoteDropRouter
from rag.rag_pipeline import answer_with_hybrid, answer_with_agentic


# ---------------------------------------------------------------------------
# How to launch the server. Adjust if your teammate's entrypoint differs.
# ---------------------------------------------------------------------------
SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=["-u", "-m", "mcp_server.server"],
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
)

memory_buffer = RollingBuffer(max_turns=10)
scratchpad = Scratchpad()
router = PromoteDropRouter()


def _get_requested_schema(params: Any) -> dict[str, Any] | None:
    schema = getattr(params, "requested_schema", None)
    if schema is None:
        schema = getattr(params, "requestedSchema", None)
    if schema is None:
        return None
    if isinstance(schema, dict):
        return schema
    if hasattr(schema, "model_dump"):
        return schema.model_dump(mode="json")
    return {"schema": str(schema)}


def _read_content_text(content: Any) -> str:
    if isinstance(content, TextContent):
        return content.text
    if hasattr(content, "text"):
        return str(content.text)
    if hasattr(content, "blob"):
        return str(content.blob)
    if hasattr(content, "resource"):
        resource = content.resource
        if hasattr(resource, "text"):
            return str(resource.text)
        if hasattr(resource, "blob"):
            return str(resource.blob)
    if hasattr(content, "model_dump"):
        return json.dumps(content.model_dump(mode="json"), indent=2)
    return str(content)


# ---------------------------------------------------------------------------
# Elicitation: called mid-tool-call when the server needs a human decision
# (e.g. approving a high-value return, confirming a controlled action).
# ---------------------------------------------------------------------------
async def elicitation_callback(
    context: ClientRequestContext,
    params: ElicitRequestParams,
) -> ElicitResult:
    print("\n" + "=" * 60)
    print("  SERVER IS ASKING FOR HUMAN INPUT (elicitation/create)")
    print("=" * 60)
    print(f"Message : {params.message}")

    requested_schema = _get_requested_schema(params)
    if requested_schema:
        print(f"Expected fields: {json.dumps(requested_schema, indent=2)}")

    answer = input("\nApprove this action? [y/n]: ").strip().lower()
    if answer != "y":
        return ElicitResult(action="decline")

    content: dict[str, Any] = {}
    schema = requested_schema or {}
    properties = schema.get("properties", {}) if isinstance(schema, dict) else getattr(schema, "properties", {})
    for field_name, field_def in properties.items():
        description = ""
        if isinstance(field_def, dict):
            description = field_def.get("description", "")
        value = input(f"  {field_name} ({description}): ")
        content[field_name] = value

    return ElicitResult(action="accept", content=content or None)


# ---------------------------------------------------------------------------
# Sampling: called when the server requests an LLM generation
# (e.g. generating a personalized apology email with sampling/createMessage).
# ---------------------------------------------------------------------------
async def sampling_callback(
    context: ClientRequestContext,
    params: CreateMessageRequestParams,
) -> CreateMessageResult:
    print("\n" + "=" * 60)
    print("  SERVER IS REQUESTING LLM SAMPLING (sampling/createMessage)")
    print("=" * 60)
    system_prompt = getattr(params, "systemPrompt", None) or getattr(params, "system_prompt", "") or ""
    print(f"System Prompt: {system_prompt}")

    prompt_text = ""
    for msg in getattr(params, "messages", []):
        content = getattr(msg, "content", None)
        prompt_text += f"\n[{getattr(msg, 'role', 'user')}]: {_read_content_text(content)}"
    print(f"Messages:\n{prompt_text}")

    reply = "Dear Customer, we apologize for the order delay. Your item is on its way."

    return CreateMessageResult(
        role="assistant",
        content=TextContent(type="text", text=reply),
        model="mock-client-llm",
        stop_reason="endTurn",
    )


# ---------------------------------------------------------------------------
# Notifications: fires whenever the server pushes tools/list_changed
# (e.g. a user's role was switched and new tools became available).
# ---------------------------------------------------------------------------
async def message_handler(message: Any) -> None:
    notification = getattr(message, "root", message)
    if isinstance(notification, ProgressNotification):
        params = getattr(notification, "params", None)
        if params is None:
            return
        progress = getattr(params, "progress", 0)
        total = getattr(params, "total", None)
        pct = f"{progress}/{total}" if total else str(progress)
        label = getattr(params, "message", "") or ""
        print(f"  [progress] {pct}  {label}")
        return

    method = getattr(notification, "method", None)
    if method == "notifications/tools/list_changed":
        print("\n>>> [notification] tools/list_changed received — refreshing tool list...")


# ---------------------------------------------------------------------------
# Helpers to print discovered capabilities/tools/resources/prompts
# ---------------------------------------------------------------------------
async def print_capabilities(session: ClientSession) -> dict[str, Any]:
    init_result = await session.initialize()
    caps = init_result.capabilities
    tools_caps = getattr(caps, "tools", None)
    resources_caps = getattr(caps, "resources", None)
    prompts_caps = getattr(caps, "prompts", None)
    logging_caps = getattr(caps, "logging", None)

    print("Server capabilities declared in initialize():")
    print(f"  tools     : {tools_caps}")
    print(f"  resources : {resources_caps}")
    print(f"  prompts   : {prompts_caps}")
    print(f"  logging   : {logging_caps}")
    return {
        "tools_list_changed": bool(getattr(tools_caps, "list_changed", None)),
        "elicitation": True,
    }


async def print_tools(session: ClientSession) -> list[str]:
    result = await session.list_tools()
    names = [t.name for t in result.tools]
    print("\nAvailable tools for the current session/role:")
    for t in result.tools:
        print(f"  - {t.name}: {t.description}")
    return names


async def print_resources(session: ClientSession) -> None:
    result = await session.list_resources()
    print("\nAvailable resources:")
    for r in result.resources:
        print(f"  - {r.name} ({r.uri})")


async def print_prompts(session: ClientSession) -> None:
    result = await session.list_prompts()
    print("\nAvailable prompts:")
    for p in result.prompts:
        args = ", ".join(a.name for a in (p.arguments or []))
        print(f"  - {p.name}({args})")


# ---------------------------------------------------------------------------
# RAG display helpers
# ---------------------------------------------------------------------------
def _print_rag_result(result: dict) -> None:
    print("\n" + "-" * 60)
    print("ANSWER:")
    print(result["answer"])
    print("-" * 60)

    if result.get("relevance_checks"):
        print("\nSelf-RAG relevance checks:")
        for check in result["relevance_checks"]:
            status = "PASS" if check.passed else "FAIL"
            print(f"  [{status}] {check.reasoning}")

    if result.get("hops"):
        print("\nAgentic RAG hops:")
        for hop in result["hops"]:
            print(f"  - sub-question: {hop.query}")
            print(f"    reasoning: {hop.reasoning}")

    support_check = result.get("support_check")
    if support_check is not None:
        status = "GROUNDED" if support_check.passed else "NOT GROUNDED"
        print(f"\nSelf-RAG support check: [{status}] {support_check.reasoning}")


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------
def remember(role: str, content: str):
    evicted = memory_buffer.add_turn(role, content)
    if evicted:
        router.route([evicted])


async def interactive_loop(session: ClientSession, capability_flags: dict[str, Any]) -> None:
    while True:
        tool_names = await print_tools(session)
        await print_resources(session)
        await print_prompts(session)

        print("\nOptions:")
        print("  1) Call a tool")
        print("  2) Read a resource")
        print("  3) Get a prompt")
        print("  4) Quit")
        print("  5) Show memory")
        print("  6) Ask a knowledge question (Hybrid RAG)")
        print("  7) Ask a multi-part question (Agentic RAG)")
        choice = input("> ").strip()

        if choice == "1":
            name = input(f"Tool name {tool_names}: ").strip()
            scratchpad.set_sub_goal(f"Call tool {name}")

            raw_args = input("Arguments as JSON (e.g. {\"order_id\": 5}) or blank: ").strip()
            arguments = json.loads(raw_args) if raw_args else {}

            if not capability_flags["tools_list_changed"]:
                print("  (note: server did NOT declare tools.listChanged=True — "
                      "notifications may not be relied on here)")

            try:
                remember("user", f"Tool request: {name} {arguments}")
                result = await session.call_tool(name, arguments)
            except Exception as exc:
                print(f"\nERROR calling tool: {exc}")
                continue
            for block in result.content:
                tool_text = _read_content_text(block)
                remember("tool", tool_text)
                print(f"\nResult:\n{tool_text}")

        elif choice == "2":
            uri = input("Resource URI (e.g. resource://return_policy): ").strip()
            try:
                result = await session.read_resource(uri)
            except Exception as exc:
                print(f"\nERROR reading resource: {exc}")
                continue
            for content in result.contents:
                print(f"\n{_read_content_text(content)}")

        elif choice == "3":
            name = input("Prompt name: ").strip()
            raw_args = input("Arguments as JSON or blank: ").strip()
            arguments = json.loads(raw_args) if raw_args else {}
            try:
                result = await session.get_prompt(name, arguments)
            except Exception as exc:
                print(f"\nERROR fetching prompt: {exc}")
                continue
            for msg in result.messages:
                print(f"\n[{msg.role}] {_read_content_text(msg.content)}")

        elif choice == "4":
            break

        elif choice == "5":
            print("\n--- Rolling Buffer ---")
            for item in memory_buffer.get_context():
                print(item)

            print("\n--- Scratchpad ---")
            print(scratchpad.snapshot())

            print("\n--- Episodic Memory ---")
            for episode in router.episodic_store:
                print(episode)

        elif choice == "6":
            query = input("Question: ").strip()
            if not query:
                continue
            scratchpad.set_sub_goal(f"Answer knowledge question via hybrid RAG: {query}")
            remember("user", f"Knowledge question: {query}")

            result = answer_with_hybrid(query)
            _print_rag_result(result)
            remember("assistant", result["answer"])

        elif choice == "7":
            query = input("Question (can be multi-part): ").strip()
            if not query:
                continue
            scratchpad.set_sub_goal(f"Answer multi-part question via agentic RAG: {query}")
            remember("user", f"Multi-part question: {query}")

            result = answer_with_agentic(query)
            _print_rag_result(result)
            remember("assistant", result["answer"])

        else:
            print("Unknown option.")


async def main() -> None:
    parser = argparse.ArgumentParser(description="MarketLoop MCP agent client")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default=os.getenv("MARKETLOOP_TRANSPORT", "stdio"),
        help="Transport to the server (default: stdio)",
    )
    parser.add_argument(
        "--server-url",
        default=os.getenv("MARKETLOOP_SERVER_URL", "http://localhost:8000/mcp"),
        help="Streamable HTTP endpoint (used with --transport http)",
    )
    args = parser.parse_args()

    async def _run_session(reads: Any, writes: Any) -> None:
        async with ClientSession(
            reads,
            writes,
            elicitation_callback=elicitation_callback,
            sampling_callback=sampling_callback,
            message_handler=message_handler,
        ) as session:
            capability_flags = await print_capabilities(session)
            await interactive_loop(session, capability_flags)

    if args.transport == "http":
        print(f"Connecting to MarketLoop server over Streamable HTTP: {args.server_url}")
        try:
            async with streamable_http_client(args.server_url) as (read, write):
                await _run_session(read, write)
        except Exception as exc:
            print(f"\nCould not reach the MarketLoop server at {args.server_url}: {exc}")
            print("Start the server in Streamable HTTP mode first, or use --transport stdio.")
    else:
        print("Connecting to MarketLoop server over stdio")
        async with stdio_client(SERVER_PARAMS) as (read, write):
            await _run_session(read, write)


if __name__ == "__main__":
    asyncio.run(main())