from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from planning_lab.algorithms.decomposition import (
    decompose_goal,
    execute_plan,
)
from planning_lab.algorithms.environment import Environment

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

    def validate_plan(
        self,
        state: GraphState,
    ) -> TransitionResult:

        plan_data = state.data.get("plan")

        if not plan_data:
            raise ValueError(
                "No decomposition plan found."
            )

        from planning_lab.models import Plan

        plan = Plan.model_validate(plan_data)

        return TransitionResult(
            next_node="execute_parallel_tasks",
            updates={
                "data": {
                    **state.data,
                    "validated_plan": plan.model_dump(),
                }
            },
        )

    def execute_parallel_tasks(
        self,
        state: GraphState,
    ) -> TransitionResult:

        from planning_lab.models import Plan

        plan_data = state.data.get("validated_plan")

        if not plan_data:
            raise ValueError(
                "No validated plan found."
            )

        plan = Plan.model_validate(plan_data)

        outputs = execute_plan(
            plan,
            self.llm,
            environment=self.environment,
        )

        return TransitionResult(
            next_node="ground_result",
            updates={
                "outputs": {
                    **state.outputs,
                    "execution": outputs,
                }
            },
        )

    def ground_result(
        self,
        state: GraphState,
    ) -> TransitionResult:

        if self.environment is None:
            return TransitionResult(
                next_node="synthesis",
            )

        execution = state.outputs.get(
            "execution",
            {},
        )

        feedback = self.environment.evaluate(
            str(execution)
        )

        if feedback.success:
            return TransitionResult(
                next_node="synthesis",
                updates={
                    "outputs": {
                        **state.outputs,
                        "environment_feedback":
                            feedback.model_dump(),
                    }
                },
            )

        return TransitionResult(
            next_node="retry_refine",
            updates={
                "outputs": {
                    **state.outputs,
                    "environment_feedback":
                        feedback.model_dump(),
                }
            },
        )

    def retry_refine(
        self,
        state: GraphState,
    ) -> TransitionResult:

        return TransitionResult(
            next_node="execute_parallel_tasks",
            updates={
                "data": {
                    **state.data,
                    "refinement_requested": True,
                }
            },
        )

    def synthesis(
        self,
        state: GraphState,
    ) -> TransitionResult:

        execution = state.outputs.get(
            "execution",
        )

        return TransitionResult(
            next_node="done",
            status="done",
            updates={
                "outputs": {
                    **state.outputs,
                    "final": execution,
                }
            },
        )

    def nodes(self) -> dict[str, Any]:
        return {
            "awaiting_input": self.awaiting_input,
            "decompose": self.decompose,
            "validate_plan": self.validate_plan,
            "execute_parallel_tasks":
                self.execute_parallel_tasks,
            "ground_result": self.ground_result,
            "retry_refine": self.retry_refine,
            "synthesis": self.synthesis,
        }