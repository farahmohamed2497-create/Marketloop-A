from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from planning_lab.algorithms.lats import lats

from state_graph.core.models import (
    GraphState,
    TransitionResult,
)

from state_graph.hitl.node import HITLNode



class LATSAndHITLGraph:
    """Graph 2: LATS search with human intervention."""

    def __init__(
        self,
        *,
        llm: BaseChatModel,
        environment: Environment,
        confidence_threshold: float = 0.70,
    ) -> None:

        self.llm = llm
        self.environment = environment
        self.confidence_threshold = confidence_threshold
        self.hitl = HITLNode()

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
            next_node="lats_search",
        )

    def lats_search(
        self,
        state: GraphState,
    ) -> TransitionResult:

        result = lats(
            state.goal,
            self.llm,
            self.environment,
        )

        return TransitionResult(
            next_node="evaluate",
            updates={
                "outputs": {
                    **state.outputs,
                    "lats_result": {
                        "success": result.success,
                        "output": result.output,
                        "best_score": result.best_score,
                        "iterations": result.iterations,
                    },
                }
            },
        )


    def hitl(
        self,
        state: GraphState,
    ) -> TransitionResult:

        feedback = state.outputs.get(
            "environment_feedback",
            {},
        )

        reason = feedback.get(
            "details",
            ["Low confidence"],
        )

        request_id = self.hitl.create_request(
            run_id=state.run_id,
            graph_name=state.graph_name,
            reason=str(reason),
            state=state.model_dump(),
        )

        return TransitionResult(
            next_node="hitl",
            status="waiting",
            updates={
                "waiting_request_id": request_id,
            },
        )

    def resume(
        self,
        state: GraphState,
    ) -> TransitionResult:

        return TransitionResult(
            next_node="lats_search",
            status="running",
            updates={
                "waiting_request_id": None,
            },
        )

    def nodes(self) -> dict[str, Any]:
        return {
            "awaiting_input": self.awaiting_input,
            "lats_search": self.lats_search,
            "hitl": self.hitl,
            "resume": self.resume,
        }