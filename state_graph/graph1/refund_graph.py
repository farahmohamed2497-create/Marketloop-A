from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from planning_lab.algorithms.lats import lats
from planning_lab.algorithms.environment import Environment
from state_graph.core.models import GraphState, TransitionResult


class RefundGraph:
    """State graph for handling refund requests.

    LATS is used to explore multiple refund decisions before the graph
    performs deterministic validation and approval routing.
    """

    def __init__(
        self,
        *,
        llm: BaseChatModel,
        environment: Environment,
    ) -> None:
        self.llm = llm
        self.environment = environment

    def awaiting_input(
        self,
        state: GraphState,
    ) -> TransitionResult:
        """Wait until a refund request is available."""

        if not state.goal.strip():
            return TransitionResult(
                next_node="awaiting_input",
                status="waiting",
            )

        return TransitionResult(
            next_node="lats",
        )

    def lats_node(
        self,
        state: GraphState,
    ) -> TransitionResult:
        """Explore and evaluate alternative refund decisions with LATS."""

        result = lats(
            task=state.goal,
            llm=self.llm,
            environment=self.environment,
        )

        updated_data = {
            **state.data,
            "lats": {
                "success": result.success,
                "output": result.output,
                "best_score": result.best_score,
                "iterations": result.iterations,
            },
        }

        return TransitionResult(
            next_node="evaluate_refund",
            updates={
                "data": updated_data,
            },
        )

    def nodes(self) -> dict[str, Any]:
        return {
            "awaiting_input": self.awaiting_input,
            "lats": self.lats_node,
        }