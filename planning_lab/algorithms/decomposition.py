from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field

from ..models import Plan
from ..models import EnvironmentFeedback
from .environment import Environment


class TaskExecutor(Protocol):
    """Adapter for executing a DAG node through an existing MCP tool."""

    def execute(self, tool_name: str, arguments: dict[str, object]) -> str: ...


PLANNER_SYSTEM = """You are a careful task-decomposition planner.

Produce a small executable DAG, not a prose checklist.

Every task must make a concrete contribution to the goal.

Independent research or analysis tasks should be parallel.

Dependencies must refer only to tasks that exist in the plan.

The plan must end with exactly one synthesis task depending on every
necessary branch.

Use short task ids such as:
t1, t2, t3, ...

Keep the task instructions concrete and executable.

For MarketLoop sales-audit requests, separate independent sales, returns,
inventory, and audit-log analysis nodes before a risk-analysis and a final
management-synthesis node. Preserve only genuine data dependencies.

Only bind a task to a tool when it needs a real system-of-record result. The
available MarketLoop tools are generate_sales_audit_report (read-only) and
update_inventory_quantity (a manager-approved inventory action).
"""


class PlannedTask(BaseModel):
    """LLM-facing schema for one decomposed task."""

    model_config = ConfigDict(extra="forbid")

    id: str
    instruction: str
    depends_on: list[str]
    tool_name: str | None = None
    tool_arguments: dict[str, object] = Field(default_factory=dict)


class GeneratedPlan(BaseModel):
    """LLM-facing schema for the generated decomposition."""

    model_config = ConfigDict(extra="forbid")

    goal: str
    tasks: list[PlannedTask]


def decompose_goal(goal: str, llm: BaseChatModel) -> Plan:
    """Decompose a goal into a validated executable DAG.

    The LLM generates the candidate plan.
    The domain Plan model remains authoritative for DAG validation,
    including duplicate IDs, missing dependencies, self-dependencies,
    and cycles.
    """

    generated = llm.with_structured_output(
        GeneratedPlan,
        method="json_schema",
    ).invoke(
        [
            ("system", PLANNER_SYSTEM),
            (
                "human",
                f"""Decompose this goal into 3-6 executable tasks:

{goal!r}

Requirements:
- Use short task ids such as t1, t2, t3.
- Dependencies may refer only to tasks in this plan.
- Independent tasks should not depend on each other unnecessarily.
- Include exactly one final synthesis task.
- The final synthesis task must depend on every branch needed for the goal.
- Bind real data collection to generate_sales_audit_report with start_date and
  end_date arguments. Bind an explicitly approved restock action only to
  update_inventory_quantity with product_id, quantity_change, and user_id.
- Preserve the supplied goal exactly in the plan's goal field.""",
            ),
        ],
        temperature=0.1,
    )

    payload = generated.model_dump()

    # The user's original goal is authoritative.
    payload["goal"] = goal

    # Plan.model_validate() performs the actual DAG validation.
    return Plan.model_validate(payload)


def execute_plan(
    plan: Plan,
    llm: BaseChatModel,
    max_workers: int = 4,
    environment: Environment | None = None,
    max_grounding_retries: int = 1,
    task_executor: TaskExecutor | None = None,
) -> dict[str, str]:
    """Execute a validated DAG batch by batch.

    Tasks in the same execution batch have no unresolved dependency
    between them and can therefore run in parallel.
    """

    outputs: dict[str, str] = {}

    for batch in plan.execution_batches():
        prompts: dict[str, str] = {}

        for task_id in batch:
            task = plan.task(task_id)

            context = "\n\n".join(
                f"OUTPUT FROM {dependency}:\n{outputs[dependency]}"
                for dependency in task.depends_on
            ) or "No prerequisite outputs."

            prompts[task_id] = f"""Overall goal:
{plan.goal}

Current task:
{task.instruction}

Prerequisite outputs:
{context}

Complete only the current task.
Be concrete and concise.
Do not invent sources."""

        with ThreadPoolExecutor(
            max_workers=min(max_workers, len(batch))
        ) as pool:
            futures = {
                (
                    pool.submit(
                        task_executor.execute,
                        plan.task(task_id).tool_name,
                        plan.task(task_id).tool_arguments,
                    )
                    if task_executor is not None and plan.task(task_id).tool_name
                    else pool.submit(
                        llm.invoke,
                        [
                            (
                                "system",
                                "You execute one node in a validated task DAG.",
                            ),
                            ("human", prompt),
                        ],
                        temperature=0.2,
                    )
                ): task_id
                for task_id, prompt in prompts.items()
            }

            for future in as_completed(futures):
                task_id = futures[future]

                completed = future.result()
                content = completed if isinstance(completed, str) else completed.content

                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError(
                        "The chat model returned an empty or unsupported response"
                    )

                result = content.strip()

                if environment is not None:
                    feedback = environment.evaluate(result)

                    retries = 0

                    while (
                            not feedback.success
                            and retries < max_grounding_retries
                    ):
                        task = plan.task(task_id)

                        feedback_text = "\n".join(
                            f"- {item}"
                            for item in feedback.details
                        ) or "- External evaluator rejected the result."

                        retry_prompt = f"""Overall goal: {plan.goal}

            Current task: {task.instruction}

            Previous attempt:
            {result}

            External environment feedback:
            Score: {feedback.score}
            Success: {feedback.success}
            Details:
            {feedback_text}

            Revise the current task output using the external feedback.
            Return only the improved result.
            """

                        response = llm.invoke(
                            [
                                (
                                    "system",
                                    "You are executing a task using grounded external feedback.",
                                ),
                                ("human", retry_prompt),
                            ],
                            temperature=0.2,
                        )

                        revised = response.content

                        if not isinstance(revised, str) or not revised.strip():
                            raise RuntimeError(
                                "The chat model returned an empty or unsupported response"
                            )

                        result = revised.strip()
                        feedback = environment.evaluate(result)

                        retries += 1

                outputs[task_id] = result

    return outputs


def final_output(
    plan: Plan,
    outputs: dict[str, str],
) -> str:
    """Return the output of the single terminal synthesis task."""

    terminals = plan.terminal_tasks()

    if len(terminals) != 1:
        raise ValueError(
            "Expected exactly one terminal synthesis task, "
            f"found {terminals}"
        )

    return outputs[terminals[0]]