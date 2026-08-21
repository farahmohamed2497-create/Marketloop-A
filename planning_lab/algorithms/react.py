from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


class ToolNotAllowedError(RuntimeError):
    """
    Raised when the model attempts to call a tool outside the
    constrained set it was given.

    This is intentionally an exception, not a silently-ignored event:
    an out-of-policy tool call must surface through
    StateGraphEngine.step's except-block as a failure ticket, not be
    swallowed and retried.
    """


@dataclass
class ToolCallRecord:
    tool_name: str
    arguments: dict[str, Any]
    result: Any


@dataclass
class ReactResult:
    success: bool
    output: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    iterations: int = 0
    confidence: float = 1.0
    escalated: bool = False


def constrained_react(
    *,
    task: str,
    llm: BaseChatModel,
    tools: dict[str, Callable[..., Any]],
    max_steps: int = 6,
    system_prompt: str | None = None,
) -> ReactResult:
    """
    Run a constrained ReAct loop.

    Unlike open-ended ReAct, the model may ONLY call the tools
    explicitly passed in `tools`. This is what makes the technique
    "constrained": the action space is small, fixed, and every action
    has a real external side effect (a carrier API call), so the
    tool set has to be locked down for compliance reasons rather than
    left open for the model to improvise.
    """
    bound_llm = llm.bind_tools(list(tools.values()))

    messages: list[Any] = [
        SystemMessage(
            content=system_prompt
            or (
                "You are a constrained shipping-support agent. "
                f"You may ONLY use these tools: {', '.join(tools.keys())}. "
                "Never invent a tool that isn't in that list. If you "
                "cannot resolve the issue confidently with these "
                "tools, call escalate_to_hitl instead of guessing."
            )
        ),
        HumanMessage(content=task),
    ]

    tool_calls: list[ToolCallRecord] = []
    escalated = False

    for step in range(max_steps):
        response: AIMessage = bound_llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return ReactResult(
                success=True,
                output=response.content,
                tool_calls=tool_calls,
                iterations=step + 1,
                confidence=_estimate_confidence(tool_calls, escalated),
                escalated=escalated,
            )

        for call in response.tool_calls:
            name = call["name"]

            if name not in tools:
                raise ToolNotAllowedError(
                    f"Model attempted to call disallowed tool: {name!r}"
                )

            result = tools[name](**call["args"])

            tool_calls.append(
                ToolCallRecord(
                    tool_name=name,
                    arguments=call["args"],
                    result=result,
                )
            )

            if name == "escalate_to_hitl":
                escalated = True

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=call["id"],
                )
            )

        if escalated:
            return ReactResult(
                success=False,
                output="Escalated to human review.",
                tool_calls=tool_calls,
                iterations=step + 1,
                confidence=0.0,
                escalated=True,
            )

    return ReactResult(
        success=False,
        output="Max steps exceeded without resolution.",
        tool_calls=tool_calls,
        iterations=max_steps,
        confidence=_estimate_confidence(tool_calls, escalated),
        escalated=escalated,
    )


def _estimate_confidence(
    tool_calls: list[ToolCallRecord],
    escalated: bool,
) -> float:
    """
    Heuristic confidence score fed into the HITL policy.

    Starts at 1.0 and loses 0.12 per tool call needed to resolve the
    issue (more calls -> less certain a clean resolution), and drops
    straight to 0.0 if the agent itself escalated.
    """
    if escalated:
        return 0.0

    penalty_per_call = 0.12
    confidence = 1.0 - (penalty_per_call * len(tool_calls))

    return max(0.0, min(1.0, confidence))
