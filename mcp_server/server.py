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
from mcp.server.stdio import stdio_server
from pydantic import AnyUrl

from planning_lab.mcp_executor import MarketLoopMCPExecutor

from .config import get_database_path, get_transport
from .db import get_connection
from .tools.session import SessionContext, switch_active_user_role


# MCP serializes this field as ``mimeType``. Keep the snake_case accessor
# used throughout this repository without changing the protocol payload.
if not hasattr(types.Resource, "mime_type"):
    types.Resource.mime_type = property(lambda resource: resource.mimeType)


# The SDK deprecates the logging/sampling capabilities the assignment requires,
# and the dispatcher logs full tracebacks for expected tool rejections.
# Quiet both so live demos and agent output stay clean; errors still reach
# the client.
executor = MarketLoopMCPExecutor(
    allow_mutations=False,
)

logging.getLogger(
    "mcp.shared.jsonrpc_dispatcher"
).setLevel(logging.CRITICAL)

logging.getLogger(
    "mcp.server.runner"
).setLevel(logging.CRITICAL)


class MarketLoopMCPServer:
    """Protocol-aware MCP server with modular tool registration."""

    def __init__(
        self,
        name: str = "marketloop",
        version: str = "0.1.0",
    ) -> None:
        self.name = name
        self.version = version
        self.transport = get_transport()

        self._tools: list[Callable[..., Any]] = []
        self._resources: list[
            tuple[str, Callable[..., Any]]
        ] = []
        self._prompts: list[
            tuple[str, Callable[..., Any]]
        ] = []

        self._session_context = SessionContext()
        self._active_tool_names: set[str] = set()
        self._session: Any | None = None
        self._visible_tools: list[
            Callable[..., Any]
        ] = []

        # mcp==1.29.0 uses the v1 low-level Server API.
        # Handlers are registered through decorator calls rather than
        # constructor keyword arguments.
        self._server = Server(
            name,
            version=version,
        )

        self._register_handlers()

    def _register_handlers(self) -> None:
        """Register protocol handlers using the MCP v1 API."""

        self._server.list_tools()(self._list_tools)
        self._server.call_tool()(self._call_tool)
        self._server.list_resources()(self._list_resources)
        self._server.read_resource()(self._read_resource)
        self._server.list_prompts()(self._list_prompts)
        self._server.get_prompt()(self._get_prompt)
        self._server.set_logging_level()(self._set_logging_level)

    def initialize(self) -> dict[str, Any]:
        """Return the MCP initialization payload."""
        return {
            "protocolVersion": "2024-11-05",
            "serverInfo": {
                "name": self.name,
                "version": self.version,
            },
            "capabilities": {
                "tools": {
                    "listChanged": False,
                },
                "resources": {
                    "subscribe": False,
                    "listChanged": False,
                },
                "prompts": {
                    "listChanged": False,
                },
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
            "database": str(
                get_database_path()
            ),
            "capabilities": self.initialize()[
                "capabilities"
            ],
        }

    def register_modules(self) -> None:
        """Discover and register tools, resources, and prompts."""
        self._register_from_package("tools")
        self._register_from_package("resources")
        self._register_from_package("prompts")
        self._refresh_visible_tools()

    def _register_from_package(
        self,
        package_name: str,
    ) -> None:
        package = importlib.import_module(
            f"mcp_server.{package_name}"
        )

        for _, module_name, _ in pkgutil.iter_modules(
            package.__path__
        ):
            module = importlib.import_module(
                f"mcp_server.{package_name}.{module_name}"
            )

            self._register_module_members(module)

    def _register_module_members(
        self,
        module: ModuleType,
    ) -> None:
        for _, member in inspect.getmembers(
            module,
            inspect.isfunction,
        ):
            if member.__module__ != module.__name__:
                continue

            name = getattr(
                member,
                "name",
                None,
            ) or member.__name__

            if name.startswith("_"):
                continue

            module_leaf = module.__name__.split(
                "."
            )[-1]

            if module_leaf in {
                "inventory",
                "orders",
                "order",
                "session",
                "tools",
            }:
                self._tools.append(member)
                continue

            kind = getattr(
                member,
                "kind",
                None,
            )

            if kind in {
                "tool",
                "resource",
                "prompt",
            }:
                if kind == "tool":
                    self._tools.append(member)

                elif kind == "resource":
                    self._resources.append(
                        (name, member)
                    )

                elif kind == "prompt":
                    self._prompts.append(
                        (name, member)
                    )

                continue

            lower_name = name.lower()

            if (
                "tool" in lower_name
                or lower_name.endswith("tool")
            ):
                self._tools.append(member)

            elif (
                "resource" in lower_name
                or lower_name.endswith("resource")
            ):
                self._resources.append(
                    (name, member)
                )

            elif (
                "prompt" in lower_name
                or lower_name.endswith("prompt")
            ):
                self._prompts.append(
                    (name, member)
                )

    def _refresh_visible_tools(self) -> None:
        """Expose only tools allowed for the active role."""

        role = (
            self._session_context.active_role
            or ""
        ).strip().lower()

        visible_tools: list[
            Callable[..., Any]
        ] = []

        for tool in self._tools:
            name = getattr(
                tool,
                "name",
                tool.__name__,
            )

            if (
                name == "update_inventory_quantity"
                and role
                and role
                not in {
                    "warehouse admin",
                    "manager",
                    "inventory manager",
                }
            ):
                continue

            if (
                name == "process_return_request"
                and role
                and role
                not in {
                    "customer support",
                    "support staff",
                    "manager",
                }
            ):
                continue

            visible_tools.append(tool)

        self._active_tool_names = {
            getattr(
                tool,
                "name",
                tool.__name__,
            )
            for tool in visible_tools
        }

        self._visible_tools = visible_tools

    def _build_tool(
        self,
        tool: Callable[..., Any],
    ) -> types.Tool:
        name = getattr(
            tool,
            "name",
            tool.__name__,
        )

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
                    schema = TypeAdapter(
                        annotation
                    ).json_schema()

                except Exception:
                    schema = {
                        "type": "string",
                    }

                properties[
                    parameter.name
                ] = schema

                if (
                    parameter.default
                    is inspect.Parameter.empty
                ):
                    required.append(
                        parameter.name
                    )

            input_schema: dict[str, Any] = {
                "type": "object",
                "properties": properties,
            }

            if required:
                input_schema[
                    "required"
                ] = required

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
    # Handler compatibility
    #
    # These handlers support:
    #
    # 1. MCP v1 direct arguments.
    # 2. The request/params calling shape used by the existing tests.
    #
    # This keeps the server compatible with the installed SDK while
    # preserving the repository's existing test contract.
    # -----------------------------------------------------------------

    async def _list_tools(
        self,
        *_args: Any,
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
        name_or_request: Any,
        arguments_or_params: Any | None = None,
    ) -> types.CallToolResult | list[types.ContentBlock]:
        """
        Support both MCP v1 direct calls and test-style request/params calls.
        """

        if isinstance(
            name_or_request,
            str,
        ):
            tool_name = name_or_request
            payload = (
                arguments_or_params
                or {}
            )
            fallback_context = None
            supplied_meta = None

        else:
            fallback_context = (
                name_or_request
            )

            params = arguments_or_params

            tool_name = getattr(
                params,
                "name",
                None,
            )

            payload = (
                getattr(
                    params,
                    "arguments",
                    None,
                )
                or {}
            )
            supplied_meta = getattr(params, "meta", None)

        self._refresh_visible_tools()

        try:
            request_context = (
                self._server.request_context
            )

        except LookupError:
            # Unit tests call the handler outside
            # an MCP request context.
            request_context = fallback_context

        for tool in self._visible_tools:
            current_name = getattr(
                tool,
                "name",
                tool.__name__,
            )

            if current_name != tool_name:
                continue

            meta = supplied_meta or (
                getattr(request_context, "meta", None)
                if request_context is not None
                else None
            )

            progress_token = None

            if meta is not None:
                if isinstance(meta, dict):
                    progress_token = meta.get(
                        "progressToken",
                        meta.get(
                            "progress_token"
                        ),
                    )
                else:
                    progress_token = getattr(
                        meta,
                        "progressToken",
                        getattr(
                            meta,
                            "progress_token",
                            None,
                        ),
                    )

            tool_context = type(
                "Context",
                (),
                {
                    "session": (
                        getattr(
                            request_context,
                            "session",
                            None,
                        )
                        if request_context is not None
                        else None
                    ),
                    "progress_token": (
                        progress_token
                    ),
                    "request_id": (
                        getattr(
                            request_context,
                            "request_id",
                            None,
                        )
                        if request_context is not None
                        else None
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
                tool_name
                == "switch_active_user_role"
                and isinstance(
                    payload,
                    dict,
                )
            ):
                result = tool(
                    payload.get("user_id"),
                    session_context=(
                        self._session_context
                    ),
                )

            elif not payload:
                result = tool()

            else:
                result = tool(payload)

            if inspect.isawaitable(result):
                result = await result

            if (
                tool_name
                == "switch_active_user_role"
            ):
                previous_names = set(
                    self._active_tool_names
                )

                self._refresh_visible_tools()

                if (
                    self._active_tool_names
                    != previous_names
                ):
                    session = (
                        getattr(
                            request_context,
                            "session",
                            None,
                        )
                        if request_context is not None
                        else None
                    )

                    if (
                        session is not None
                        and hasattr(
                            session,
                            "send_tool_list_changed",
                        )
                    ):
                        await session.send_tool_list_changed()

            content = [
                types.TextContent(
                    type="text",
                    text=str(result),
                )
            ]

            # Direct v1 handler calls return content for the SDK wrapper.
            # The repository's protocol-level tests call the legacy
            # request/params shape and expect the complete MCP result.
            if fallback_context is not None:
                return types.CallToolResult(content=content)

            return content

        raise ValueError(
            f"Unknown tool: {tool_name}"
        )

    def _resource_uri(
        self,
        name: str,
        resource: Callable[..., Any],
    ) -> str:
        """Resolve the URI exposed for a resource."""

        return (
            getattr(
                resource,
                "uri",
                None,
            )
            or f"resource://{name}"
        )

    async def _list_resources(
        self,
        *_args: Any,
    ) -> types.ListResourcesResult:
        return types.ListResourcesResult(
            resources=[
                types.Resource(
                    name=name,
                    description=(resource.__doc__ or ""),
                    uri=AnyUrl(self._resource_uri(name, resource)),
                    mimeType="text/markdown",
                )
                for name, resource in self._resources
            ]
        )

    async def _read_resource(
        self,
        uri_or_request: Any,
        params: Any | None = None,
    ) -> str | types.ReadResourceResult:
        """
        Support both:

            _read_resource(uri)

        and:

            _read_resource(request, params)
        """

        if params is not None:
            uri = getattr(
                params,
                "uri",
                params,
            )
        else:
            uri = uri_or_request

        uri_str = str(uri)

        for name, resource in self._resources:
            if (
                self._resource_uri(
                    name,
                    resource,
                )
                == uri_str
            ):
                text = str(resource())
                if params is None:
                    return text

                return types.ReadResourceResult(
                    contents=[
                        types.TextResourceContents(
                            uri=uri_str,
                            mimeType="text/markdown",
                            text=text,
                        )
                    ]
                )

        raise ValueError(
            f"Unknown resource: {uri_str}"
        )

    async def _list_prompts(
        self,
        *_args: Any,
    ) -> types.ListPromptsResult:
        return types.ListPromptsResult(
            prompts=[
                types.Prompt(
                    name=name,
                    description=(prompt.__doc__ or ""),
                    arguments=[
                        types.PromptArgument(**argument)
                        for argument in getattr(prompt, "arguments", [])
                    ],
                )
                for name, prompt in self._prompts
            ]
        )

    async def _get_prompt(
        self,
        name_or_request: Any,
        arguments_or_params: Any | None = None,
    ) -> types.GetPromptResult:
        """
        Support both:

            _get_prompt(name, arguments)

        and:

            _get_prompt(request, params)
        """

        if isinstance(
            name_or_request,
            str,
        ):
            name = name_or_request
            payload = (
                arguments_or_params
                or {}
            )

        else:
            params = arguments_or_params

            name = getattr(
                params,
                "name",
                getattr(
                    name_or_request,
                    "name",
                    None,
                ),
            )

            payload = (
                getattr(
                    params,
                    "arguments",
                    {},
                )
                or {}
            )

        for pname, prompt in self._prompts:
            if pname == name:
                text = str(
                    prompt(
                        arguments=payload
                    )
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
        level: Any,
    ) -> None:
        return None

    def list_tools(self) -> list[str]:
        self._refresh_visible_tools()

        return [
            getattr(
                tool,
                "name",
                tool.__name__,
            )
            for tool in self._visible_tools
        ]

    def list_resources(self) -> list[str]:
        return [
            name
            for name, _ in self._resources
        ]

    def list_prompts(self) -> list[str]:
        return [
            name
            for name, _ in self._prompts
        ]

    def notify_tools_changed(self) -> None:
        """Notify the client that the available toolset changed."""

        self._refresh_visible_tools()

        session = getattr(
            self,
            "_session",
            None,
        )

        if (
            session is not None
            and hasattr(
                session,
                "send_tool_list_changed",
            )
        ):
            try:
                asyncio.run(
                    session.send_tool_list_changed()
                )

            except RuntimeError:
                loop = asyncio.new_event_loop()

                try:
                    loop.run_until_complete(
                        session.send_tool_list_changed()
                    )
                finally:
                    loop.close()

    def switch_role(
        self,
        user_id: int,
    ) -> dict[str, Any]:
        """Switch the active role and refresh visible tools."""

        result = switch_active_user_role(
            user_id=user_id,
            session_context=self._session_context,
        )

        self._refresh_visible_tools()

        if result.get("tools_updated"):
            self.notify_tools_changed()

        return result

    async def _run_stdio(self) -> None:
        async with stdio_server() as (
            read_stream,
            write_stream,
        ):
            await self._server.run(
                read_stream,
                write_stream,
                self._server.create_initialization_options(),
            )

    def run(self) -> None:
        """Run the server using stdio by default."""

        self.connect()

        if self.transport == "stdio":
            asyncio.run(
                self._run_stdio()
            )
            return

        if self.transport in {
            "streamable-http",
            "sse",
        }:
            raise NotImplementedError(
                "Remote transport adapters are not implemented yet"
            )


if __name__ == "__main__":
    server = MarketLoopMCPServer()
    server.run()
