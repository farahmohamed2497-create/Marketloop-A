"""MarketLoop MCP server entrypoint."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
import pkgutil
import warnings
from types import ModuleType
from typing import Any, Callable

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPDeprecationWarning

from .config import get_database_path, get_transport
from .db import get_connection
from .tools.session import SessionContext, switch_active_user_role

# The SDK deprecates the logging/sampling capabilities the assignment requires,
# and the dispatcher logs full tracebacks for expected tool rejections. Quiet
# both so live demos and agent output stay clean; errors still reach the client.
warnings.filterwarnings("ignore", category=MCPDeprecationWarning)
logging.getLogger("mcp.shared.jsonrpc_dispatcher").setLevel(logging.CRITICAL)
logging.getLogger("mcp.server.runner").setLevel(logging.CRITICAL)


class MarketLoopMCPServer:
    """A protocol-aware MCP server with capability negotiation and modular registration."""

    def __init__(self, name: str = "marketloop", version: str = "0.1.0") -> None:
        self.name = name
        self.version = version
        self.transport = get_transport()
        self._tools: list[Callable[..., Any]] = []
        self._resources: list[tuple[str, Callable[..., Any]]] = []
        self._prompts: list[tuple[str, Callable[..., Any]]] = []
        self._session_context = SessionContext()
        self._active_tool_names: set[str] = set()
        self._session: Any | None = None
        self._server = Server(
            name=name,
            version=version,
            on_list_tools=self._list_tools,
            on_call_tool=self._call_tool,
            on_list_resources=self._list_resources,
            on_read_resource=self._read_resource,
            on_list_prompts=self._list_prompts,
            on_get_prompt=self._get_prompt,
            on_set_logging_level=self._set_logging_level,
        )

    def initialize(self) -> dict[str, Any]:
        """Return the initialize payload for MCP capability negotiation."""
        return {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": self.name, "version": self.version},
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
                "prompts": {"listChanged": False},
                "logging": {},
            },
        }

    def connect(self) -> dict[str, Any]:
        """Prepare the server for a transport connection."""
        self.register_modules()
        connection = get_connection()
        connection.close()
        return {
            "transport": self.transport,
            "database": str(get_database_path()),
            "capabilities": self.initialize()["capabilities"],
        }

    def register_modules(self) -> None:
        """Discover and register tools, resources, and prompts from the package tree."""
        self._register_from_package("tools")
        self._register_from_package("resources")
        self._register_from_package("prompts")
        self._refresh_visible_tools()

    def _register_from_package(self, package_name: str) -> None:
        package = importlib.import_module(f"mcp_server.{package_name}")
        for _, module_name, _ in pkgutil.iter_modules(package.__path__):
            module = importlib.import_module(f"mcp_server.{package_name}.{module_name}")
            self._register_module_members(module)

    def _register_module_members(self, module: ModuleType) -> None:
        for _, member in inspect.getmembers(module, inspect.isfunction):
            if member.__module__ != module.__name__:
                continue
            name = getattr(member, "name", None) or member.__name__
            if name.startswith("_"):
                continue
            if module.__name__.split(".")[-1] in {"inventory", "orders", "order", "session", "tools"}:
                self._tools.append(member)
                continue
            kind = getattr(member, "kind", None)
            if kind in {"tool", "resource", "prompt"}:
                if kind == "tool":
                    self._tools.append(member)
                elif kind == "resource":
                    self._resources.append((name, member))
                elif kind == "prompt":
                    self._prompts.append((name, member))
                continue
            if "tool" in name.lower() or name.lower().endswith("tool"):
                self._tools.append(member)
            elif "resource" in name.lower() or name.lower().endswith("resource"):
                self._resources.append((name, member))
            elif "prompt" in name.lower() or name.lower().endswith("prompt"):
                self._prompts.append((name, member))

    def _refresh_visible_tools(self) -> None:
        """Expose only the tools appropriate for the current active role."""
        role = (self._session_context.active_role or "").strip().lower()
        visible_tools = []
        for tool in self._tools:
            name = getattr(tool, "name", tool.__name__)
            if name == "update_inventory_quantity" and role and role not in {"warehouse admin", "manager", "inventory manager"}:
                continue
            if name == "process_return_request" and role and role not in {"customer support", "support staff", "manager"}:
                continue
            visible_tools.append(tool)
        self._active_tool_names = {getattr(tool, "name", tool.__name__) for tool in visible_tools}
        self._visible_tools = visible_tools

    def _build_tool(self, tool: Callable[..., Any]) -> types.Tool:
        name = getattr(tool, "name", tool.__name__)
        description = tool.__doc__ or ""
        return types.Tool(
            name=name,
            description=description,
            input_schema={"type": "object", "properties": {}},
        )

    async def _list_tools(self, _context: Any, _params: Any = None) -> types.ListToolsResult:
        self._refresh_visible_tools()
        return types.ListToolsResult(tools=[self._build_tool(tool) for tool in self._visible_tools])

    async def _call_tool(self, _context: Any, params: Any) -> types.CallToolResult:
        tool_name = params.name
        payload = None
        if hasattr(params, "arguments") and params.arguments is not None:
            payload = params.arguments
        elif isinstance(params, dict):
            payload = params
        self._refresh_visible_tools()
        for tool in self._visible_tools:
            if getattr(tool, "name", tool.__name__) == tool_name:
                meta = getattr(params, "meta", None)
                progress_token = None
                if isinstance(meta, dict):
                    progress_token = meta.get("progressToken", meta.get("progress_token"))
                tool_context = type(
                    "Context",
                    (),
                    {
                        "session": getattr(_context, "session", None),
                        "progress_token": progress_token,
                        "request_id": getattr(_context, "request_id", None),
                    },
                )()
                if tool_name in {"generate_sales_audit_report", "process_return_request", "generate_delay_apology"}:
                    result = await tool(payload, context=tool_context)
                elif tool_name == "switch_active_user_role" and isinstance(payload, dict):
                    result = tool(payload.get("user_id"), session_context=self._session_context)
                elif payload is None:
                    result = tool()
                else:
                    result = tool(payload)
                if inspect.isawaitable(result):
                    result = await result

                # Role switches mutate the visible tool set at runtime. Refresh it
                # and push tools/list_changed so clients re-discover the tools
                # instead of polling or guessing.
                if tool_name == "switch_active_user_role":
                    previous_names = set(self._active_tool_names)
                    self._refresh_visible_tools()
                    if self._active_tool_names != previous_names:
                        session = getattr(_context, "session", None)
                        if session is not None and hasattr(session, "send_tool_list_changed"):
                            await session.send_tool_list_changed()

                return types.CallToolResult(content=[types.TextContent(type="text", text=str(result))])
        raise ValueError(f"Unknown tool: {tool_name}")

    def _resource_uri(self, name: str, resource: Callable[..., Any]) -> str:
        """Resolve the URI a resource is exposed under, defaulting to a per-name URI."""
        return getattr(resource, "uri", None) or f"resource://{name}"

    async def _list_resources(self, _context: Any, _params: Any = None) -> types.ListResourcesResult:
        return types.ListResourcesResult(
            resources=[
                types.Resource(
                    name=name,
                    description=resource.__doc__ or "",
                    uri=self._resource_uri(name, resource),
                    mime_type="text/markdown",
                )
                for name, resource in self._resources
            ]
        )

    async def _read_resource(self, _context: Any, params: Any) -> types.ReadResourceResult:
        for name, resource in self._resources:
            if self._resource_uri(name, resource) == params.uri:
                return types.ReadResourceResult(
                    contents=[
                        types.TextResourceContents(
                            uri=params.uri,
                            mime_type="text/markdown",
                            text=str(resource()),
                        )
                    ]
                )
        raise ValueError(f"Unknown resource: {params.uri}")

    async def _list_prompts(self, _context: Any, _params: Any = None) -> types.ListPromptsResult:
        return types.ListPromptsResult(
            prompts=[
                types.Prompt(
                    name=name,
                    description=prompt.__doc__ or "",
                    arguments=[
                        types.PromptArgument(**argument)
                        for argument in getattr(prompt, "arguments", [])
                    ],
                )
                for name, prompt in self._prompts
            ]
        )

    async def _get_prompt(self, _context: Any, params: Any) -> types.GetPromptResult:
        for name, prompt in self._prompts:
            if name == params.name:
                arguments = getattr(params, "arguments", None) or {}
                text = str(prompt(arguments=arguments))
                return types.GetPromptResult(
                    messages=[
                        types.PromptMessage(
                            role="user",
                            content=types.TextContent(type="text", text=text),
                        )
                    ]
                )
        raise ValueError(f"Unknown prompt: {params.name}")

    async def _set_logging_level(self, _context: Any, _params: Any) -> types.EmptyResult:
        return types.EmptyResult()

    def list_tools(self) -> list[str]:
        self._refresh_visible_tools()
        return [getattr(tool, "name", tool.__name__) for tool in self._visible_tools]

    def list_resources(self) -> list[str]:
        return [name for name, _ in self._resources]

    def list_prompts(self) -> list[str]:
        return [name for name, _ in self._prompts]

    def notify_tools_changed(self) -> None:
        """Notify the client that the available toolset changed."""
        self._refresh_visible_tools()
        session = getattr(self, "_session", None)
        if session is not None and hasattr(session, "send_tool_list_changed"):
            try:
                import asyncio
                asyncio.run(session.send_tool_list_changed())
            except RuntimeError:
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(session.send_tool_list_changed())
                finally:
                    loop.close()

    def switch_role(self, user_id: int) -> dict[str, Any]:
        """Switch the active role and refresh the visible toolset."""
        result = switch_active_user_role(user_id=user_id, session_context=self._session_context)
        self._refresh_visible_tools()
        if result.get("tools_updated"):
            self.notify_tools_changed()
        return result

    async def _run_stdio(self) -> None:
        async with stdio_server() as (read_stream, write_stream):
            await self._server.run(read_stream, write_stream, self._server.create_initialization_options())

    def run(self) -> None:
        """Run the server using stdio by default."""
        self.connect()
        if self.transport == "stdio":
            asyncio.run(self._run_stdio())
            return
        if self.transport in {"streamable-http", "sse"}:
            raise NotImplementedError("Remote transport adapters are not implemented yet")


if __name__ == "__main__":
    server = MarketLoopMCPServer()
    server.run()
