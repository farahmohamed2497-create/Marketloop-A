from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict
from typing import Callable, Optional


class DynamicDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    done: bool
    next_task: str


def _default_executor(llm: BaseChatModel, goal: str, task: str, observation: str) -> str:
    """
    Fallback executor: asks the LLM to describe what it would do.
    TODO: replace with a real executor once the sub-task domain is defined
    (e.g. one that calls the actual MCP tool matching `task`).
    """
    response = llm.invoke([
        ("system", "Execute the next adaptive sub-task using the observations provided."),
        ("human", f"Goal: {goal}\nNext task: {task}\nPrior observations:\n{observation}"),
    ], temperature=0.2)
    result = response.content
    if not isinstance(result, str) or not result.strip():
        raise RuntimeError("The chat model returned an empty or unsupported response")
    return result.strip()


def dynamic_decomposition(
    goal: str,
    llm: BaseChatModel,
    max_steps: int = 4,
    executor: Optional[Callable[[BaseChatModel, str, str, str], str]] = None,
) -> list[tuple[str, str]]:
    """
    `executor` is the function that actually carries out each sub-task once
    decided. Defaults to an LLM-only stand-in; pass a real executor (one that
    calls MCP tools / the database) once the sub-task domain is defined.
    """
    executor = executor or _default_executor
    history: list[tuple[str, str]] = []

    for step in range(max_steps):
        observation = "\n".join(f"{task}: {result}" for task, result in history) or "None"
        decision = llm.with_structured_output(
            DynamicDecision,
            method="json_schema",
        ).invoke([
            ("system", "You are an adaptive planner. Use prior observations before deciding what comes next."),
            ("human", f"""Goal: {goal}
Completed work and observations:
{observation}

Decide the single best next task. Set done to true only when the goal is met.
When done is true, use an empty string for next_task."""),
        ], temperature=0.1)

        if decision.done:
            break

        task = decision.next_task.strip()
        if not task:
            raise ValueError(f"Dynamic planner omitted next_task at step {step + 1}")

        result = executor(llm, goal, task, observation)
        history.append((task, result))

    return history


# ---------------------------------------------------------------------------
# Model provider: Groq, replacing the toolkit's default
# ---------------------------------------------------------------------------

def get_llm() -> BaseChatModel:
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0)