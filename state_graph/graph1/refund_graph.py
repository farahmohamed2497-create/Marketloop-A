from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from planning_lab.algorithms.environment import Environment
from planning_lab.algorithms.lats import lats
from state_graph.core.models import GraphState, TransitionResult
from state_graph.hitl.node import HITLNode
from state_graph.hitl.policy import requires_human_intervention


class RefundGraph:
    """State graph for handling refund requests.

    LATS explores alternative refund decisions. When the resulting
    confidence, refund amount, or policy status requires human review,
    the graph pauses through the persistent HITL node.
    """

    def __init__(
        self,
        *,
        llm: BaseChatModel,
        environment: Environment,
    ) -> None:
        self.llm = llm
        self.environment = environment
        self.hitl = HITLNode()

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
            next_node="policy_grounding",
        )

    def policy_grounding(self, state: GraphState) -> TransitionResult:
        """RAG addition: retrieve the return policy before selecting a refund.

        The retrieved answer is persisted so LATS and a later human reviewer
        make the same decision from the same policy evidence after recovery.
        """
        from rag.rag_pipeline import answer_with_hybrid

        retrieval = answer_with_hybrid(
            f"Return/refund policy evidence needed for this request: {state.goal}"
        )
        return TransitionResult(
            next_node="lats",
            updates={"data": {**state.data, "policy_context": retrieval["answer"]}},
        )

    def lats_node(
        self,
        state: GraphState,
    ) -> TransitionResult:
        """Run LATS and route to HITL when policy requires human review."""

        result = lats(
            task=f"{state.goal}\n\nRelevant company policy:\n{state.data.get('policy_context', '')}",
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

        refund_amount = updated_data.get(
            "refund_amount"
        )

        policy_violation = bool(
            updated_data.get(
                "policy_violation",
                False,
            )
        )

        if requires_human_intervention(
            score=result.best_score,
            refund_amount=refund_amount,
            policy_violation=policy_violation,
        ):
            paused_state = state.model_copy(
                update={
                    "data": updated_data,
                }
            )

            return self.hitl.pause(
                paused_state,
                reason=(
                    "Refund requires human review "
                    "because the configured HITL policy "
                    "was triggered."
                ),
            )

        return TransitionResult(
            next_node="evaluate_refund",
            updates={
                "data": updated_data,
            },
        )

    def evaluate_refund(self, state: GraphState) -> TransitionResult:
        """Wait for the physical return before the refund can be finalized."""
        if not state.data.get("return_received", False):
            return TransitionResult(next_node="awaiting_return", status="waiting")
        return TransitionResult(
            next_node="done",
            status="done",
            updates={"outputs": {**state.outputs, "resolution": "Refund approved after return inspection."}},
        )

    def awaiting_return(self, state: GraphState) -> TransitionResult:
        """External warehouse receipt loops back to evaluation, never to LATS."""
        if not state.data.get("return_received", False):
            return TransitionResult(next_node="awaiting_return", status="waiting")
        return TransitionResult(next_node="evaluate_refund")

    def nodes(self) -> dict[str, Any]:
        return {
            "awaiting_input": self.awaiting_input,
            "policy_grounding": self.policy_grounding,
            "lats": self.lats_node,
            "evaluate_refund": self.evaluate_refund,
            "awaiting_return": self.awaiting_return,
        }
