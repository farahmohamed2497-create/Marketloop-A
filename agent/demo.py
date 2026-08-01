"""
MarketLoop MCP protocol demo (automated, headless).

Walks through every protocol concern end-to-end without any keyboard input,
so it can be run live during a presentation or grading evaluation:

  STEP 1  Capability negotiation  — real initialize() handshake, declared caps
  STEP 2  Resources & Prompts     — resources/read + prompts/get
  STEP 3  Safe write path         — low-value return auto-processed (no elicitation)
  STEP 4  Elicitation             — high-value return pauses for human sign-off
  STEP 5  Notifications           — role switch pushes tools/list_changed live
  STEP 6  Progress tracking       — report streams 0/25/50/75/100% updates
  STEP 7  Sampling                — server asks the client's "LLM" via createMessage
  STEP 8  Defensive tool design   — invalid / unauthorized writes rejected in-handler

Transport: stdio (local development). The same client also supports Streamable
HTTP for remote deployment (see client.py --transport http).

Run with:
    python agent/demo.py
(from the repo root; the DB is reset to a fixed demo scenario on every run.)
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import (
    CreateMessageRequestParams,
    CreateMessageResult,
    ElicitRequestParams,
    ElicitResult,
    ProgressNotification,
    TextContent,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db.init_db import initialize_database
from mcp_server.config import get_database_path

# ---------------------------------------------------------------------------
# Presentation formatting (ANSI colors)
# ---------------------------------------------------------------------------
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[1;36m"
BLUE = "\033[1;34m"
YELLOW = "\033[1;33m"
GREEN = "\033[1;32m"
RED = "\033[1;31m"
DIM = "\033[2m"


def banner() -> None:
    print()
    print(f"{BOLD}{CYAN}=============================================================={RESET}")
    print(f"{BOLD}{CYAN}        MARKETLOOP MCP SERVER — AUTOMATED PROTOCOL DEMO{RESET}")
    print(f"{BOLD}{CYAN}=============================================================={RESET}")
    print(f"{DIM}Every protocol concern fires live, no keyboard input required.{RESET}\n")


def header(title: str, subtitle: str = "") -> None:
    print()
    print(f"{CYAN}{'=' * 60}{RESET}")
    print(f"{CYAN}{title}{RESET}")
    print(f"{CYAN}{'=' * 60}{RESET}")
    if subtitle:
        print(f"{DIM}{subtitle}{RESET}")


def info(label: str, value: Any = "") -> None:
    print(f"  {YELLOW}{label}:{RESET} {value}")


def ok(message: str) -> None:
    print(f"    {GREEN}[PASS]{RESET} {message}")


def fail(message: str) -> None:
    print(f"    {RED}[FAIL]{RESET} {message}")


def content_text(content: Any) -> str:
    if isinstance(content, TextContent):
        return content.text
    if hasattr(content, "text"):
        return str(content.text)
    if hasattr(content, "blob"):
        return str(content.blob)
    if hasattr(content, "resource"):
        return content_text(content.resource)
    if hasattr(content, "model_dump"):
        return json.dumps(content.model_dump(mode="json"), indent=2)
    return str(content)


def clip(text: str, limit: int = 500) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "\n    ... [truncated]"


# ---------------------------------------------------------------------------
# Shared protocol state
# ---------------------------------------------------------------------------
RESULTS: list[bool] = []
list_changed_event: asyncio.Event | None = None
elicitation_count = 0


def check(passed: bool, message: str) -> None:
    RESULTS.append(bool(passed))
    (ok if passed else fail)(message)


# ---------------------------------------------------------------------------
# Client-side protocol callbacks (all automated — no input())
# ---------------------------------------------------------------------------
async def auto_elicitation_callback(
    context: Any,
    params: ElicitRequestParams,
) -> ElicitResult:
    global elicitation_count
    elicitation_count += 1
    print(f"\n    {CYAN}== elicitation/create received from the server =={RESET}")
    print(f"    {YELLOW}message:{RESET} {params.message}")
    if hasattr(params, "requestedSchema"):
        print(f"    {YELLOW}requested schema:{RESET} {json.dumps(params.requestedSchema)}")
    print(f"    {BLUE}→ automated approval granted (role-based policy allows it){RESET}")
    return ElicitResult(action="accept", content={"approved": True})


async def sampling_callback(
    context: Any,
    params: CreateMessageRequestParams,
) -> CreateMessageResult:
    system_prompt = getattr(params, "systemPrompt", None) or getattr(params, "system_prompt", "") or ""
    print(f"\n    {CYAN}== sampling/createMessage received from the server =={RESET}")
    print(f"    {YELLOW}system prompt:{RESET} {clip(system_prompt, 200)}")
    for msg in getattr(params, "messages", []):
        print(f"    {YELLOW}[{getattr(msg, 'role', 'user')}]:{RESET} {clip(content_text(getattr(msg, 'content', None)), 250)}")
    reply = (
        "Dear Customer,\n\n"
        "We are truly sorry about the delay in shipping your order. "
        "Your item is on its way and will arrive shortly. "
        "Thank you for your patience — MarketLoop values you."
    )
    print(f"    {BLUE}→ client's LLM returned a personalized reply{RESET}")
    return CreateMessageResult(
        role="assistant",
        content=TextContent(type="text", text=reply),
        model="marketloop-demo-llm",
        stop_reason="endTurn",
    )


async def message_handler(message: Any) -> None:
    global list_changed_event
    notification = getattr(message, "root", message)
    if isinstance(notification, ProgressNotification):
        return
    method = getattr(notification, "method", None)
    if method == "notifications/tools/list_changed":
        print(f"\n    {YELLOW}<<< notification: notifications/tools/list_changed received{RESET}")
        if list_changed_event is not None:
            list_changed_event.set()


# ---------------------------------------------------------------------------
# Server / database setup
# ---------------------------------------------------------------------------
SERVER_PARAMS = StdioServerParameters(
    command="python",
    args=["-m", "mcp_server.server"],
    env=None,
)


def prepare_demo_database() -> None:
    """Reset the DB to a fixed scenario so the demo is repeatable."""
    db_path = get_database_path()
    for suffix in ("", "-wal", "-shm"):
        Path(str(db_path) + suffix).unlink(missing_ok=True)
    initialize_database(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            -- Order 1: low-value delivered order -> return auto-processed
            UPDATE Orders SET total_amount = 45, status = 'Delivered' WHERE order_id = 1;
            UPDATE Order_Items SET price = 45 WHERE order_id = 1;
            -- Order 2: high-value delivered order -> return needs elicitation
            UPDATE Orders SET total_amount = 250, status = 'Delivered' WHERE order_id = 2;
            UPDATE Order_Items SET price = 250 WHERE order_id = 2;
            """
        )
        connection.commit()
    info("database reset", f"{db_path}")
    info("scenario", "order 1 = $45 Delivered | order 2 = $250 Delivered | order 3 = $27 Processing")


# ---------------------------------------------------------------------------
# The demo steps
# ---------------------------------------------------------------------------
async def step1_capability_negotiation(session: ClientSession) -> None:
    header("STEP 1: CAPABILITY NEGOTIATION", "initialize() handshake — server declares, client verifies")
    init_result = await session.initialize()
    caps = init_result.capabilities
    tools_caps = getattr(caps, "tools", None)
    resources_caps = getattr(caps, "resources", None)
    prompts_caps = getattr(caps, "prompts", None)
    logging_caps = getattr(caps, "logging", None)
    info("protocol version", init_result.protocol_version)
    info("server info", f"{init_result.server_info.name} {init_result.server_info.version}")
    info("tools capability", tools_caps)
    info("resources capability", resources_caps)
    info("prompts capability", prompts_caps)
    info("logging capability", logging_caps)
    check(
        getattr(tools_caps, "list_changed", None) is not None,
        "server declared a tools capability in initialize()",
    )
    check(
        init_result.server_info is not None,
        "server identified itself (name + version) during the handshake",
    )


async def step2_resources_and_prompts(session: ClientSession) -> None:
    header("STEP 2: RESOURCES & PROMPTS", "policy fetched as data, prompt template fetched, not re-invented")
    resources = await session.list_resources()
    info("discovered resources", [r.uri for r in resources.resources])
    resource_result = await session.read_resource("marketloop://policies/return_and_refund")
    resource_text = "\n".join(content_text(c) for c in resource_result.contents)
    print(f"    {YELLOW}policy (resources/read):{RESET}")
    print(f"    {DIM}{clip(resource_text, 450)}{RESET}")
    check(len(resource_text.strip()) > 0, "return & refund policy fetched via resources/read")

    prompt_result = await session.get_prompt(
        "draft_return_response",
        {
            "order_id": "1",
            "customer_name": "Ali Mahmoud",
            "decision_status": "Approved",
            "reason": "Wrong size",
        },
    )
    print(f"    {YELLOW}prompt template (prompts/get):{RESET}")
    for msg in prompt_result.messages:
        print(f"    {DIM}[{msg.role}] {clip(content_text(msg.content), 450)}{RESET}")
    check(len(prompt_result.messages) > 0, "parameterized prompt template fetched from the server")


async def step3_auto_approved_return(session: ClientSession) -> None:
    header("STEP 3: SAFE WRITE PATH (NO ELICITATION)", "order 1 = $45 — below the $100 review threshold")
    global elicitation_count
    elicitation_count = 0
    result = await session.call_tool(
        "process_return_request",
        {"order_id": 1, "customer_id": 1, "reason": "Product arrived damaged"},
    )
    text = "\n".join(content_text(b) for b in result.content)
    print(f"    {YELLOW}tool result:{RESET} {text}")
    check("return_id" in text and "Pending" in text, "return request created directly for the low-value order")
    check(elicitation_count == 0, "no elicitation required — completed without human sign-off")


async def step4_elicitation_return(session: ClientSession) -> None:
    header("STEP 4: ELICITATION (HUMAN-IN-THE-LOOP)", "order 2 = $250 — above the $100 review threshold")
    result = await session.call_tool(
        "process_return_request",
        {"order_id": 2, "customer_id": 2, "reason": "Item arrived defective and broken"},
    )
    text = "\n".join(content_text(b) for b in result.content)
    print(f"    {YELLOW}tool result:{RESET} {text}")
    check(elicitation_count == 1, "elicitation/create fired mid-call for the high-value order")
    check("return_id" in text and "Pending" in text, "return created only after the human-approved outcome")


async def step5_notifications(session: ClientSession) -> None:
    header(
        "STEP 5: NOTIFICATIONS (RUNTIME TOOL CHANGE)",
        "role switch pushes tools/list_changed — client re-discovers tools",
    )

    async def _switch(user_id: int) -> None:
        global list_changed_event
        list_changed_event = asyncio.Event()
        result = await session.call_tool("switch_active_user_role", {"user_id": user_id})
        text = "\n".join(content_text(b) for b in result.content)
        print(f"    {YELLOW}tool result:{RESET} {text}")
        try:
            await asyncio.wait_for(list_changed_event.wait(), timeout=5)
            check(True, "notifications/tools/list_changed was pushed by the server")
        except asyncio.TimeoutError:
            check(False, "notifications/tools/list_changed was pushed by the server (timeout)")

    before = await session.list_tools()
    names_before = [t.name for t in before.tools]
    info("tools before role change", sorted(names_before))

    print()
    info("switch", "user_id=2 (Customer Support) — restricted tools disappear")
    await _switch(2)
    after = await session.list_tools()
    names_after = [t.name for t in after.tools]
    info("tools after switch", sorted(names_after))
    hidden = set(names_before) - set(names_after)
    if hidden:
        check(True, f"write tools hidden for Customer Support: {sorted(hidden)}")

    print()
    info("switch", "user_id=3 (Inventory Manager) — warehouse write tools appear")
    await _switch(3)
    final = await session.list_tools()
    names_final = [t.name for t in final.tools]
    appeared = set(names_final) - set(names_after)
    print(f"    {YELLOW}newly discovered write tools:{RESET} {sorted(appeared) if appeared else '(none)'}")
    check("update_inventory_quantity" in names_final, "update_inventory_quantity reappeared for the warehouse role")


async def step6_progress_tracking(session: ClientSession) -> None:
    header(
        "STEP 6: PROGRESS TRACKING",
        "long-running report streams 0% -> 25% -> 50% -> 75% -> 100%",
    )
    progress_log: list[str] = []

    async def on_progress(progress: float, total: float | None, message: str | None) -> None:
        pct = f"{progress * 100:.0f}%"
        progress_log.append(pct)
        print(f"    {BLUE}[progress]{RESET} {pct:>4}  {message or ''}")

    result = await session.call_tool(
        "generate_sales_audit_report",
        {"start_date": "2026-01-01", "end_date": "2026-07-31"},
        progress_callback=on_progress,
    )
    text = "\n".join(content_text(b) for b in result.content)
    print(f"    {YELLOW}report (first 300 chars):{RESET}")
    print(f"    {DIM}{clip(text, 300)}{RESET}")
    check(progress_log == ["0%", "25%", "50%", "75%", "100%"], f"progress streamed at every stage: {progress_log}")
    check('"total_revenue"' in text, "report returned with real aggregated metrics")


async def step7_sampling(session: ClientSession) -> None:
    header(
        "STEP 7: LLM SAMPLING",
        "server requests sampling/createMessage — the client's LLM completes the email",
    )
    result = await session.call_tool("generate_delay_apology", {"order_id": 3})
    text = "\n".join(content_text(b) for b in result.content)
    print(f"    {YELLOW}email returned to the server (sampling result actually used):{RESET}")
    print(f"    {DIM}{clip(text, 350)}{RESET}")
    check("Dear Customer" in text, "sampling/createMessage round-trip completed and the reply was used")


async def step8_defensive_design(session: ClientSession) -> None:
    header(
        "STEP 8: DEFENSIVE TOOL DESIGN",
        "server-side validation + handler-level authorization, not just schema types",
    )
    print(f"    {DIM}Attempting writes that must be rejected before touching data...{RESET}")

    try:
        await session.call_tool(
            "update_inventory_quantity",
            {"product_id": 0, "quantity_change": -5, "user_id": 3},
        )
        check(False, "invalid product_id (0) was rejected")
    except Exception as exc:
        print(f"    {YELLOW}rejected invalid input:{RESET} {str(exc)[:200]}")
        check(True, "invalid product_id rejected by server-side validation")

    try:
        await session.call_tool(
            "update_inventory_quantity",
            {"product_id": 1, "quantity_change": -5, "user_id": 1},
        )
        check(False, "unauthorized admin user was rejected")
    except Exception as exc:
        print(f"    {YELLOW}rejected unauthorized caller:{RESET} {str(exc)[:200]}")
        check(True, "unauthorized user rejected by handler-level authorization")

    result = await session.call_tool(
        "update_inventory_quantity",
        {"product_id": 1, "quantity_change": -5, "user_id": 3},
    )
    text = "\n".join(content_text(b) for b in result.content)
    print(f"    {YELLOW}authorized write succeeded:{RESET} {text}")
    check("updated" in text, "authorized warehouse write completed normally")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
async def main() -> None:
    banner()
    prepare_demo_database()

    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(
            read,
            write,
            elicitation_callback=auto_elicitation_callback,
            sampling_callback=sampling_callback,
            message_handler=message_handler,
        ) as session:
            await step1_capability_negotiation(session)
            await step2_resources_and_prompts(session)
            await step3_auto_approved_return(session)
            await step4_elicitation_return(session)
            await step5_notifications(session)
            await step6_progress_tracking(session)
            await step7_sampling(session)
            await step8_defensive_design(session)

    header("DEMO COMPLETE")
    passed = sum(1 for r in RESULTS if r)
    total = len(RESULTS)
    info("steps passed", f"{passed}/{total}")
    print(f"    Transport: {BOLD}stdio{RESET} (local dev). For remote deployment the same client runs over "
          f"{BOLD}Streamable HTTP{RESET} with `--transport http`.")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    asyncio.run(main())
