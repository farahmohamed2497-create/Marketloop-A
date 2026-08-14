from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field

from planning_lab.mcp_executor import MarketLoopMCPExecutor


class DynamicDecision(BaseModel):
    """One adaptive planning decision."""

    model_config = ConfigDict(extra="forbid")

    done: bool = False
    instruction: str = Field(min_length=5)
    tool_name: str | None = None
    tool_arguments: dict[str, Any] = Field(default_factory=dict)


@dataclass
class DynamicStep:
    """Evidence from one adaptive step."""

    instruction: str
    result: str
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None


# The only MCP tool the dynamic planner is allowed to bind a step to for
# real system-of-record data. Any other tool_name the model invents is
# treated as an unsupported binding and downgraded to a reasoning step
# instead of crashing the run.
SUPPORTED_TOOL_NAME = "generate_sales_audit_report"

# The only keys the real generate_sales_audit_report tool accepts. Extra
# keys the model invents (product_id, include_pending_returns, etc.) are
# dropped before the call instead of raising a pydantic validation error.
SUPPORTED_TOOL_ARGUMENTS = frozenset({"start_date", "end_date"})


DYNAMIC_SYSTEM = """
You are the adaptive decomposition controller for the MarketLoop Sales Audit Agent.

Your job is NOT to write a final answer immediately.

You must follow this loop:

1. Decide the NEXT concrete sub-task.
2. If the sub-task needs a real system-of-record result, bind it to a real MCP tool.
3. Observe the result of that sub-task.
4. Use that result to decide what should happen next.
5. Stop only when the original goal is completely satisfied.

Return one decision at a time.

For Sales Audit requests:

- The ONLY tool you may bind a step to is generate_sales_audit_report.
  It accepts exactly two arguments: start_date and end_date.
- There is no separate inventory, low-stock, or returns tool. If you need
  low-stock, out-of-stock, or return information, do NOT invent a new
  tool name. Instead, either call generate_sales_audit_report (its
  result already covers sales, returns, and inventory) or leave
  tool_name unset and reason over the observations you already have.
- Do not invent database values.
- Do not invent products, orders, revenue, units, returns, or inventory.
- Do not use update_inventory_quantity for read-only audit requests.
- Only use update_inventory_quantity when the user explicitly requests
  an inventory mutation and provides the required approved arguments.
- A tool result is an observation and must be treated as evidence for later steps.
- Later sub-tasks must be based on the observed result, not on assumptions.

Return a structured decision with:
- done
- instruction
- tool_name
- tool_arguments
"""


def _validate_tool_decision(
    goal: str,
    decision: DynamicDecision,
) -> DynamicDecision:
    """
    Prevent the adaptive planner from turning a read-only sales audit
    into a database mutation or hallucinated data collection step.
    """
    goal_lower = goal.lower()

    if "sales audit" in goal_lower:
        if decision.tool_name == "update_inventory_quantity":
            raise ValueError(
                "Read-only Sales Audit cannot use update_inventory_quantity."
            )

        if (
            not decision.done
            and decision.tool_name is None
            and any(
                token in decision.instruction.lower()
                for token in (
                    "gather sales data",
                    "retrieve sales data",
                    "get sales data",
                    "collect sales data",
                )
            )
        ):
            raise ValueError(
                "Sales-data collection must use "
                "generate_sales_audit_report instead of LLM-generated facts."
            )

    return decision


def _sanitize_tool_decision(decision: DynamicDecision) -> DynamicDecision:
    """
    Make an unsupported tool_name / tool_arguments combination safe
    instead of letting it crash the run.

    - Unknown tool names (e.g. "read_low_stock_products", "inventory",
      "get_low_stock_products") are cleared, so the step falls back to
      the normal reasoning path instead of raising ValueError.
    - Extra argument keys the model invents for the supported tool
      (e.g. "product_id", "include_pending_returns") are dropped so the
      real tool's strict pydantic schema doesn't reject the call.
    """
    if decision.tool_name is None:
        return decision

    if decision.tool_name != SUPPORTED_TOOL_NAME:
        # Unsupported/invented tool name: downgrade to a reasoning-only
        # step instead of raising.
        return decision.model_copy(
            update={"tool_name": None, "tool_arguments": {}}
        )

    clean_arguments = {
        key: value
        for key, value in decision.tool_arguments.items()
        if key in SUPPORTED_TOOL_ARGUMENTS
    }

    if clean_arguments != decision.tool_arguments:
        return decision.model_copy(
            update={"tool_arguments": clean_arguments}
        )

    return decision


def _execute_step(
    decision: DynamicDecision,
    llm: BaseChatModel,
    executor: MarketLoopMCPExecutor,
    goal: str,
    observed_history: list[DynamicStep],
) -> str:
    """
    Execute exactly one adaptive step.

    Real MCP tools are executed through MarketLoopMCPExecutor.
    Only reasoning tasks without a system-of-record lookup are handled by
    the LLM itself.
    """
    if decision.tool_name is not None:
        # decision has already been sanitized by _sanitize_tool_decision,
        # so tool_name here is guaranteed to be SUPPORTED_TOOL_NAME with
        # only allowed argument keys.
        return executor.execute(
            decision.tool_name,
            decision.tool_arguments,
        )

    previous_context = "\n\n".join(
        f"STEP: {item.instruction}\nRESULT:\n{item.result}"
        for item in observed_history
    ) or "No previous observations."

    response = llm.invoke(
        [
            (
                "system",
                """
You are executing one reasoning-only sub-task in an adaptive planning loop.

Use ONLY the supplied observations.
Do not invent facts.
Do not invent database results.
Return only the result of the current reasoning task.
""",
            ),
            (
                "human",
                f"""Overall goal:
{goal}

Observed previous steps:
{previous_context}

Current reasoning task:
{decision.instruction}
""",
            ),
        ],
        temperature=0.2,
    )

    content = response.content

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(
            "The dynamic reasoning step returned an empty response."
        )

    return content.strip()


def dynamic_decomposition(
    goal: str,
    llm: BaseChatModel,
    max_steps: int = 4,
    *,
    task_executor: MarketLoopMCPExecutor | None = None,
) -> list[tuple[str, str]]:
    """
    Interleaved dynamic decomposition.

    The next sub-task is generated only after observing the result of
    the previous sub-task.

    Real MarketLoop MCP tools are executed through MarketLoopMCPExecutor.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be positive")

    executor = task_executor or MarketLoopMCPExecutor(
        allow_mutations=False
    )

    history: list[DynamicStep] = []

    for _ in range(max_steps):
        observed_context = "\n\n".join(
            f"STEP {index + 1}: {item.instruction}\n"
            f"RESULT:\n{item.result}"
            for index, item in enumerate(history)
        ) or "- No previous observations."

        decision = llm.with_structured_output(
            DynamicDecision,
            method="function_calling",
        ).invoke(
            [
                ("system", DYNAMIC_SYSTEM),
                (
                    "human",
                    f"""Original goal:
{goal}

Observed results so far:
{observed_context}

Decide ONLY the next step.

Rules:
- If another real database/MCP lookup is required, use the appropriate tool.
- For sales-audit data retrieval, use:
  generate_sales_audit_report

When using generate_sales_audit_report, provide:
{{
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD"
}}

Do not invent values that a tool should provide.

If the original goal is fully satisfied, set:
done=true
and provide a concise final synthesis instruction.
""",
                ),
            ],
            temperature=0.1,
        )

        decision = _validate_tool_decision(
            goal,
            decision,
        )

        # Downgrade any unsupported/invented tool_name or extra argument
        # keys instead of letting them raise and abort the run.
        decision = _sanitize_tool_decision(decision)

        if decision.done:
            final_instruction = decision.instruction

            if history:
                context = "\n\n".join(
                    f"STEP {index + 1}: {item.instruction}\n"
                    f"RESULT:\n{item.result}"
                    for index, item in enumerate(history)
                )
            else:
                context = "- No previous observations."

            response = llm.invoke(
                [
                    (
                        "system",
                        """
You are the final synthesis step of a dynamic planning loop.

Use ONLY the observed results.
Do not invent missing facts.
Return the final answer for the original goal.
""",
                    ),
                    (
                        "human",
                        f"""Original goal:
{goal}

Observed results:
{context}

Final synthesis instruction:
{final_instruction}
""",
                    ),
                ],
                temperature=0.2,
            )

            content = response.content

            if not isinstance(content, str) or not content.strip():
                raise RuntimeError(
                    "The final dynamic synthesis returned an empty response."
                )

            history.append(
                DynamicStep(
                    instruction=final_instruction,
                    result=content.strip(),
                )
            )

            break

        result = _execute_step(
            decision=decision,
            llm=llm,
            executor=executor,
            goal=goal,
            observed_history=history,
        )

        history.append(
            DynamicStep(
                instruction=decision.instruction,
                result=result,
                tool_name=decision.tool_name,
                tool_arguments=decision.tool_arguments,
            )
        )

    return [
        (item.instruction, item.result)
        for item in history
    ]