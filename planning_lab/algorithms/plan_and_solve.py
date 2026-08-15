from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field


class SolvePlan(BaseModel):
    """Structured plan produced before the solve phase."""

    model_config = ConfigDict(extra="forbid")

    steps: list[str] = Field(min_length=1, max_length=6)


@dataclass
class PlanAndSolveResult:
    """Evidence from both phases of Plan-and-Solve."""

    question: str
    plan: list[str]
    solution: str


def plan_and_solve(
    question: str,
    llm: BaseChatModel,
    context: str | None = None,
) -> str:
    """Solve a reasoning-heavy subtask using explicit Plan-and-Solve.

    The method has two explicit phases:
    1. Generate one ordered plan.
    2. Execute that plan sequentially in one pass.

    No branching or alternative-plan search is performed.
    """

    context_text = context.strip() if context and context.strip() else "No external context provided."

    planned = llm.with_structured_output(
        SolvePlan,
        method=getattr(llm, "structured_output_method", "json_schema"),
    ).invoke(
        [
            (
                "system",
    """You are the planning phase of a Plan-and-Solve agent.

Return ONLY valid JSON in this format:
{
  "steps": [
    "step 1",
    "step 2"
  ]
}

Create ONE concise ordered plan.

The plan must:
- contain concrete executable reasoning steps;
- preserve the order required by dependencies;
- use the supplied evidence when available;
- avoid alternative branches;
- avoid inventing facts or sources.

Do not solve the task yet.""",
            ),
            (
                "human",
                f"""Task:
{question}

Available MCP/tool context:
{context_text}

Create the ordered plan only.""",
            ),
        ],
        temperature=0.1,
    )

    if not planned.steps:
        raise RuntimeError("The planning phase returned an empty plan.")

    plan_text = "\n".join(
        f"{index}. {step.strip()}"
        for index, step in enumerate(planned.steps, start=1)
        if step.strip()
    )

    if not plan_text:
        raise RuntimeError("The planning phase returned no usable steps.")

    solved = llm.invoke(
        [
            (
                "system",
                """You are the solve phase of a Plan-and-Solve agent.

Execute the supplied plan exactly once and in order.

Rules:
- Do not create alternative branches.
- Do not replace the plan with a new plan.
- Use the supplied MCP/tool context as evidence.
- Do not invent missing facts.
- Return only the completed answer.""",
            ),
            (
                "human",
                f"""Task:
{question}

Available MCP/tool context:
{context_text}

Plan:
{plan_text}

Execute the plan step by step and produce the final answer.""",
            ),
        ],
        temperature=0.2,
    )

    if not isinstance(solved.content, str) or not solved.content.strip():
        raise RuntimeError(
            "The solve phase returned an empty or unsupported response."
        )

    return solved.content.strip()
