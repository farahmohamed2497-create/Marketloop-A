"""
MarketLoop MCP client (agent).

Connects to mcp_server/server.py over stdio and demonstrates every protocol
concern required by the assignment:

  - Capability negotiation : checks the server's declared capabilities from
                              initialize() before relying on them.
  - Notifications          : reacts live to tools/list_changed pushed by the
                              server when a user's role changes.
  - Elicitation            : answers elicitation/create requests from the
                              server (e.g. sign-off on a return/refund).
  - Resources              : lists + reads resources (policy docs, reports).
  - Prompts                : lists + fetches parameterized prompt templates.
  - Progress tracking      : shows live progress for long-running tools.

Run with:
    python agent/client.py
(from the repo root, with the mcp_server package importable, i.e. run
 `pip install -e .` first or run from a directory where `mcp_server` is on
 the Python path)
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.context import RequestContext
from mcp.types import (
    ElicitRequestParams,
    ElicitResult,
    ProgressNotification,
    TextContent,
)

# ---------------------------------------------------------------------------
# How to launch the server. Adjust if your teammate's entrypoint differs.
# ---------------------------------------------------------------------------
SERVER_PARAMS = StdioServerParameters(
    command="python",
    args=["-m", "mcp_server.server"],
    env=None,
)


# ---------------------------------------------------------------------------
# Elicitation: called mid-tool-call when the server needs a human decision
# (e.g. approving a high-value return, confirming a controlled action).
# ---------------------------------------------------------------------------
async def elicitation_callback(
    context: RequestContext[ClientSession, Any],
    params: ElicitRequestParams,
) -> ElicitResult:
    print("\n" + "=" * 60)
    print("  SERVER IS ASKING FOR HUMAN INPUT (elicitation/create)")
    print("=" * 60)
    print(f"Message : {params.message}")
    if params.requestedSchema:
        print(f"Expected fields: {json.dumps(params.requestedSchema, indent=2)}")

    answer = input("\nApprove this action? [y/n]: ").strip().lower()
    if answer != "y":
        return ElicitResult(action="decline")

    # If the server asked for structured data, collect it field by field.
    content: dict[str, Any] = {}
    schema = params.requestedSchema or {}
    for field_name, field_def in schema.get("properties", {}).items():
        value = input(f"  {field_name} ({field_def.get('description', '')}): ")
        content[field_name] = value

    return ElicitResult(action="accept", content=content or None)


# ---------------------------------------------------------------------------
# Notifications: fires whenever the server pushes tools/list_changed
# (e.g. a user's role was switched and new tools became available).
# ---------------------------------------------------------------------------
async def message_handler(message: Any) -> None:
    # Progress updates for long-running tools (report generation, etc.)
    if isinstance(message, ProgressNotification):
        params = message.root.params if hasattr(message, "root") else message.params
        pct = f"{params.progress}/{params.total}" if params.total else str(params.progress)
        label = getattr(params, "message", "") or ""
        print(f"  [progress] {pct}  {label}")
        return

    # Generic notification dispatch — catch tools/list_changed specifically.
    method = getattr(getattr(message, "root", message), "method", None)
    if method == "notifications/tools/list_changed":
        print("\n>>> [notification] tools/list_changed received — refreshing tool list...")
        # The caller re-fetches tools right after this fires; see main loop.


# ---------------------------------------------------------------------------
# Helpers to print discovered capabilities/tools/resources/prompts
# ---------------------------------------------------------------------------
async def print_capabilities(session: ClientSession) -> dict[str, Any]:
    init_result = await session.initialize()
    caps = init_result.capabilities
    print("Server capabilities declared in initialize():")
    print(f"  tools     : {caps.tools}")
    print(f"  resources : {caps.resources}")
    print(f"  prompts   : {caps.prompts}")
    print(f"  logging   : {caps.logging}")
    return {
        "tools_list_changed": bool(caps.tools and caps.tools.listChanged),
        "elicitation": True,  # negotiated via client capabilities, not shown in server payload
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
# Interactive loop
# ---------------------------------------------------------------------------
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
        choice = input("> ").strip()

        if choice == "1":
            name = input(f"Tool name {tool_names}: ").strip()
            raw_args = input("Arguments as JSON (e.g. {\"order_id\": 5}) or blank: ").strip()
            arguments = json.loads(raw_args) if raw_args else {}

            # tools/list_changed is only meaningful if the server actually
            # declared support for it — this is the capability check in action.
            if not capability_flags["tools_list_changed"]:
                print("  (note: server did NOT declare tools.listChanged=True — "
                      "notifications may not be relied on here)")

            result = await session.call_tool(name, arguments)
            for block in result.content:
                if isinstance(block, TextContent):
                    print(f"\nResult:\n{block.text}")

        elif choice == "2":
            uri = input("Resource URI (e.g. resource://return_policy): ").strip()
            result = await session.read_resource(uri)
            for content in result.contents:
                print(f"\n{content.text}")

        elif choice == "3":
            name = input("Prompt name: ").strip()
            raw_args = input("Arguments as JSON or blank: ").strip()
            arguments = json.loads(raw_args) if raw_args else {}
            result = await session.get_prompt(name, arguments)
            for msg in result.messages:
                print(f"\n[{msg.role}] {msg.content.text}")

        elif choice == "4":
            break
        else:
            print("Unknown option.")


async def main() -> None:
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(
            read,
            write,
            elicitation_callback=elicitation_callback,
            message_handler=message_handler,
        ) as session:
            capability_flags = await print_capabilities(session)
            await interactive_loop(session, capability_flags)


if __name__ == "__main__":
    asyncio.run(main())
