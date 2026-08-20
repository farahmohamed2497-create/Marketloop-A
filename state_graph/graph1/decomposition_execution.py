from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from planning_lab.algorithms.decomposition import decompose_goal
from state_graph.core.models import (
    GraphState,
    TransitionResult,
)


class DecompositionExecutionGraph:
    """Graph 1: decomposition, execution, grounding and refinement."""

    def __init__(
        self,
        *,
        llm: BaseChatModel,
        environment: Environment | None = None,
    ) -> None:
        self.llm = llm
        self.environment = environment

    def awaiting_input(
        self,
        state: GraphState,
    ) -> TransitionResult:

        if not state.goal.strip():
            return TransitionResult(
                next_node="awaiting_input",
                status="waiting",
            )

        return TransitionResult(
            next_node="decompose",
        )

    def decompose(
        self,
        state: GraphState,
    ) -> TransitionResult:

        plan = decompose_goal(
            state.goal,
            self.llm,
        )

        return TransitionResult(
            next_node="validate_plan",
            updates={
                "data": {
                    **state.data,
                    "plan": plan.model_dump(),
                }
            },
        )



    def nodes(self) -> dict[str, Any]:
        return {
            "awaiting_input": self.awaiting_input,
            "decompose": self.decompose,
        }