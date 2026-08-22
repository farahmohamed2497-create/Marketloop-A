"""Escalated return-dispute retention workflow.

LLM-call additions: Tree of Thoughts (choose a retention argument before
proposing an offer) + Constrained ReAct (execute only whitelisted
dispute-handling tools). Never finalizes a resolution on its own: the
outcome is written only after the customer accepts the offer (a real
external wait) or a compliance admin approves it through HITL.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from planning_lab.algorithms.react import constrained_react
from planning_lab.algorithms.tree_of_thoughts import tree_of_thoughts
from state_graph.core.models import GraphState, TransitionResult
from state_graph.hitl.node import HITLNode
from state_graph.hitl.policy import dispute_requires_compliance_review

from .task_queue import AdminTaskQueue, DatabaseAdminTaskQueue
from .tools import check_return_dispute, propose_retention_offer, sync_dispute_resolution


def escalate_to_hitl(*, reason: str) -> dict[str, Any]:
    """STUB tool the agent calls to signal it cannot resolve the dispute alone.

    Mirrors graph2's escalate_to_hitl: it only records the request inside
    the ReAct transcript. The actual pause + persistence happens in
    DisputeGraph.dispute_react once the ReAct loop returns.
    """
    return {"escalation_requested": True, "reason": reason}


class DisputeGraph:
    """Never applies a dispute resolution without a real customer accept
    or an explicit compliance-admin decision."""

    def __init__(self, *, llm: BaseChatModel, task_queue: AdminTaskQueue | None = None) -> None:
        self.llm = llm
        self.hitl = HITLNode()
        self.task_queue = task_queue or DatabaseAdminTaskQueue()

    def awaiting_input(self, state: GraphState) -> TransitionResult:
        return TransitionResult(
            next_node="retention_strategy" if state.goal.strip() else "awaiting_input",
            status=None if state.goal.strip() else "waiting",
        )

    def retention_strategy(self, state: GraphState) -> TransitionResult:
        """Tree-of-Thoughts: search over competing retention arguments
        (partial refund, discount on next order, executive escalation)
        before ReAct proposes a concrete offer, since which argument
        leads changes whether the customer accepts and whether
        compliance will need to review it."""
        candidates = tree_of_thoughts(
            problem=state.goal,
            llm=self.llm,
            depth=2,
            beam_width=2,
            search_strategy="bfs",
        )
        best = candidates[0] if candidates else None
        strategy_summary = {
            "strategy": best.state if best else "no viable strategy found",
            "score": best.score if best else 0.0,
            "rationale": best.rationale if best else "",
        }
        return TransitionResult(
            next_node="dispute_react",
            updates={"data": {**state.data, "retention_strategy": strategy_summary}},
        )

    def dispute_react(self, state: GraphState) -> TransitionResult:
        pending_request_id = state.data.get("hitl_request_id")
        if pending_request_id:
            return self._apply_admin_decision(state, pending_request_id)

        if state.data.get("react_done"):
            # ReAct already ran once; we're only back here because the
            # customer's response has arrived. Finalize, don't re-run it.
            return self._finalize(state, state.data.get("customer_response"))

        result = constrained_react(
            task=state.goal,
            llm=self.llm,
            tools={
                "check_return_dispute": check_return_dispute,
                "propose_retention_offer": propose_retention_offer,
                "escalate_to_hitl": escalate_to_hitl,
            },
            system_prompt=(
                "You investigate an escalated return dispute where the customer is "
                "threatening a chargeback or legal action. You may check the dispute "
                "and propose a retention offer, but you must never finalize the "
                "outcome yourself. Escalate to compliance when the customer raises a "
                "legal threat or you are not confident in the offer."
            ),
        )

        retention_offer_value = float(state.data.get("retention_offer_value", 0) or 0)
        legal_threat = bool(state.data.get("legal_threat", False))
        policy_violation = bool(state.data.get("policy_violation", False))

        summary = {
            "success": result.success,
            "output": result.output,
            "confidence": result.confidence,
            "iterations": result.iterations,
        }

        if result.escalated or dispute_requires_compliance_review(
            confidence=result.confidence,
            retention_offer_value=retention_offer_value,
            legal_threat=legal_threat,
            policy_violation=policy_violation,
        ):
            paused_state = state.model_copy(update={"data": {**state.data, "react": summary}})
            pause = self.hitl.pause(
                paused_state,
                reason="Dispute requires compliance review before an offer can be finalized.",
            )
            request_id = pause.updates["waiting_request_id"]
            persisted_data = {**paused_state.data, "hitl_request_id": request_id}
            self.task_queue.enqueue_hitl(
                request_id=request_id,
                run_id=state.run_id,
                graph_name=state.graph_name,
                state=paused_state.model_copy(update={"data": persisted_data}).model_dump(mode="json"),
            )
            return TransitionResult(
                next_node="dispute_react",
                status="waiting",
                updates={"data": persisted_data, "waiting_request_id": request_id},
            )

        # No compliance review needed yet -- the offer still needs the
        # customer's answer, which is a real external wait, not HITL.
        return TransitionResult(
            next_node="awaiting_customer_response",
            status="waiting",
            updates={"data": {**state.data, "react": summary, "react_done": True}},
        )

    def awaiting_customer_response(self, state: GraphState) -> TransitionResult:
        if state.data.get("customer_response") is None:
            return TransitionResult(next_node="awaiting_customer_response", status="waiting")
        return TransitionResult(next_node="dispute_react")

    def _finalize(self, state: GraphState, customer_response: str | None) -> TransitionResult:
        decision = "Approved" if customer_response == "accept" else "Rejected"
        sync_result = sync_dispute_resolution(
            return_id=int(state.data["return_id"]),
            decision=decision,
            resolution_note=state.data.get("react", {}).get("output", ""),
            simulate_audit_failure=bool(state.data.get("simulate_audit_failure", False)),
        )
        return TransitionResult(
            next_node="done",
            status="done",
            updates={"outputs": {"resolution": sync_result}},
        )

    def _apply_admin_decision(self, state: GraphState, request_id: str) -> TransitionResult:
        request = self.hitl.get_request(request_id)
        if request is None or request["decision"] is None:
            return TransitionResult(next_node="dispute_react", status="waiting", updates={"waiting_request_id": request_id})

        decision = request["decision"]
        sync_result = sync_dispute_resolution(
            return_id=int(state.data["return_id"]),
            decision="Approved" if decision == "approve" else "Rejected",
            resolution_note=f"Compliance decision: {decision}",
            simulate_audit_failure=bool(state.data.get("simulate_audit_failure", False)),
        )
        return TransitionResult(
            next_node="done",
            status="done",
            updates={
                "data": {**state.data, "hitl_decision": decision},
                "outputs": {"resolution": sync_result, "hitl_decision": decision},
            },
        )

    def nodes(self) -> dict[str, Any]:
        return {
            "awaiting_input": self.awaiting_input,
            "retention_strategy": self.retention_strategy,
            "dispute_react": self.dispute_react,
            "awaiting_customer_response": self.awaiting_customer_response,
        }