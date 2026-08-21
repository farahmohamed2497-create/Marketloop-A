"""MarketLoop MCP server entrypoint."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
import pkgutil
from types import ModuleType
from typing import Any, Callable

from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.stdio import stdio_server

from planning_lab.mcp_executor import MarketLoopMCPExecutor
from .config import get_database_path, get_transport
from .db import get_connection
from .tools.session import SessionContext, switch_active_user_role

# The SDK deprecates the logging/sampling capabilities the assignment requires,
# and the dispatcher logs full tracebacks for expected tool rejections. Quiet
# both so live demos and agent output stay clean; errors still reach the client.

executor = MarketLoopMCPExecutor(
    allow_mutations=False,
)


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
        # MCP 1.x registers handlers with decorators.  Passing callbacks to
        # ``Server(...)`` was supported by an older SDK API and makes the
        # server fail during construction with current 1.x releases.
        self._server = Server(name, version=version)
        self._register_protocol_handlers()

    def _register_protocol_handlers(self) -> None:
        """Adapt the project's handlers to the MCP 1.x decorator API."""

        @self._server.list_tools()
        async def list_tools_handler() -> types.ListToolsResult:
            return await self._list_tools()

        @self._server.call_tool(validate_input=False)
        async def call_tool_handler(
            tool_name: str,
            arguments: dict[str, Any],
        ) -> types.CallToolResult:
            params = type(
                "ToolParams",
                (),
                {"name": tool_name, "arguments": arguments, "meta": None},
            )()
            return await self._call_tool(self._server.request_context, params)

        @self._server.list_resources()
        async def list_resources_handler() -> types.ListResourcesResult:
            return await self._list_resources(None, None)

        @self._server.read_resource()
        async def read_resource_handler(uri: Any) -> list[ReadResourceContents]:
            result = await self._read_resource(None, type("ResourceParams", (), {"uri": uri})())
            return [
                ReadResourceContents(
                    content=content.text,
                    mime_type=content.mime_type,
                )
                for content in result.contents
            ]

        @self._server.list_prompts()
        async def list_prompts_handler() -> types.ListPromptsResult:
            return await self._list_prompts(None, None)

        @self._server.get_prompt()
        async def get_prompt_handler(
            prompt_name: str,
            arguments: dict[str, str] | None,
        ) -> types.GetPromptResult:
            params = type(
                "PromptParams",
                (),
                {"name": prompt_name, "arguments": arguments or {}},
            )()
            return await self._get_prompt(None, params)

        @self._server.set_logging_level()
        async def set_logging_level_handler(_level: Any) -> None:
            await self._set_logging_level(None, None)

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

    def _build_tool(
            self,
            tool: Callable[..., Any],
    ) -> types.Tool:
        name = getattr(tool, "name", tool.__name__)
        description = tool.__doc__ or ""

        try:
            from pydantic import TypeAdapter

            signature = inspect.signature(tool)

            properties: dict[str, Any] = {}
            required: list[str] = []

            for parameter in signature.parameters.values():
                if parameter.name in {
                    "context",
                    "session_context",
                }:
                    continue

                annotation = parameter.annotation

                if annotation is inspect.Parameter.empty:
                    continue

                try:
                    schema = TypeAdapter(annotation).json_schema()
                except Exception:
                    schema = {"type": "string"}

                properties[parameter.name] = schema

                if (
                        parameter.default
                        is inspect.Parameter.empty
                ):
                    required.append(parameter.name)

            input_schema: dict[str, Any] = {
                "type": "object",
                "properties": properties,
            }

            if required:
                input_schema["required"] = required

        except Exception:
            input_schema = {
                "type": "object",
                "properties": {},
            }

        return types.Tool(
            name=name,
            description=description,
            inputSchema=input_schema,
        )

    # -----------------------------------------------------------------
    # Handlers below are registered via `self._server.<method>()(handler)`,
    # which is the decorator-based (v1) low-level `mcp.server.Server` API.
    # In that API handlers receive the *raw protocol arguments* directly
    # (name, arguments, uri, level, ...) — there is no `_context` parameter.
    # Session/request-id access happens through the ambient
    # `self._server.request_context`, not through a handler argument.
    # -----------------------------------------------------------------

    async def _list_tools(
            self,
            _context: Any = None,
            _params: Any | None = None,
    ) -> types.ListToolsResult:
        self._refresh_visible_tools()

        return types.ListToolsResult(
            tools=[
                self._build_tool(tool)
                for tool in self._visible_tools
            ]
        )

    async def _call_tool(
            self,
            context: Any,
            params: Any,
    ) -> types.CallToolResult:
        tool_name = params.name
        payload = params.arguments or {}

        self._refresh_visible_tools()

        request_context = context

        for tool in self._visible_tools:
            if getattr(tool, "name", tool.__name__) != tool_name:
                continue

            meta = getattr(params, "meta", None)

            progress_token = None

            if meta is not None:
                if isinstance(meta, dict):
                    progress_token = meta.get(
                        "progressToken",
                        meta.get("progress_token"),
                    )
                else:
                    progress_token = getattr(
                        meta,
                        "progressToken",
                        None,
                    )

            tool_context = type(
                "Context",
                (),
                {
                    "session": getattr(
                        request_context,
                        "session",
                        None,
                    ),
                    "progress_token": progress_token,
                    "request_id": getattr(
                        request_context,
                        "request_id",
                        None,
                    ),
                },
            )()

            if tool_name in {
                "generate_sales_audit_report",
                "process_return_request",
                "generate_delay_apology",
            }:
                result = await tool(
                    payload,
                    context=tool_context,
                )

            elif (
                    tool_name == "switch_active_user_role"
                    and isinstance(payload, dict)
            ):
                result = tool(
                    payload.get("user_id"),
                    session_context=self._session_context,
                )

            elif payload is None:
                result = tool()

            else:
                result = tool(payload)

            if inspect.isawaitable(result):
                result = await result

            if tool_name == "switch_active_user_role":
                previous_names = set(
                    self._active_tool_names
                )

                self._refresh_visible_tools()

                if self._active_tool_names != previous_names:
                    session = getattr(
                        request_context,
                        "session",
                        None,
                    )

                    if (
                            session is not None
                            and hasattr(
                        session,
                        "send_tool_list_changed",
                    )
                    ):
                        await session.send_tool_list_changed()

            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=str(result),
                    )
                ]
            )

        raise ValueError(
            f"Unknown tool: {tool_name}"
        )

    def _resource_uri(self, name: str, resource: Callable[..., Any]) -> str:
        """Resolve the URI a resource is exposed under, defaulting to a per-name URI."""
        return getattr(resource, "uri", None) or f"resource://{name}"

    async def _list_resources(
            self,
            _context: Any = None,
            _params: Any | None = None,
    ) -> types.ListResourcesResult:
        return types.ListResourcesResult(
            resources=[
                types.Resource(
                    name=name,
                    description=resource.__doc__ or "",
                    uri=self._resource_uri(
                        name,
                        resource,
                    ),
                    mime_type="text/markdown",
                )
                for name, resource in self._resources
            ]
        )

    async def _read_resource(
            self,
            _context: Any,
            params: Any,
    ) -> types.ReadResourceResult:
        uri = str(params.uri)

        for name, resource in self._resources:
            if self._resource_uri(name, resource) == uri:
                return types.ReadResourceResult(
                    contents=[
                        types.TextResourceContents(
                            uri=uri,
                            mime_type="text/markdown",
                            text=str(resource()),
                        )
                    ]
                )

        raise ValueError(
            f"Unknown resource: {uri}"
        )

    async def _list_prompts(
            self,
            _context: Any = None,
            _params: Any | None = None,
    ) -> types.ListPromptsResult:
        return types.ListPromptsResult(
            prompts=[
                types.Prompt(
                    name=name,
                    description=prompt.__doc__ or "",
                    arguments=[
                        types.PromptArgument(**argument)
                        for argument in getattr(
                            prompt,
                            "arguments",
                            [],
                        )
                    ],
                )
                for name, prompt in self._prompts
            ]
        )

    async def _get_prompt(
            self,
            _context: Any,
            params: Any,
    ) -> types.GetPromptResult:
        name = params.name
        arguments = params.arguments or {}

        for pname, prompt in self._prompts:
            if pname == name:
                text = str(
                    prompt(arguments=arguments)
                )

                return types.GetPromptResult(
                    messages=[
                        types.PromptMessage(
                            role="user",
                            content=types.TextContent(
                                type="text",
                                text=text,
                            ),
                        )
                    ]
                )

        raise ValueError(
            f"Unknown prompt: {name}"
        )

    async def _set_logging_level(
            self,
            _context: Any,
            _params: Any,
    ) -> types.EmptyResult:
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
