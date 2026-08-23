"""Graph 3: LLM #2 (Constrained ReAct) and its durable HITL workflow."""
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
    """Allowed ReAct signal; this graph owns the actual pause and persistence."""
    return {"escalation_requested": True, "reason": reason}


class DisputeGraph:
    def __init__(self, *, llm: BaseChatModel, task_queue: AdminTaskQueue | None = None) -> None:
        self.llm = llm
        self.hitl = HITLNode()
        self.task_queue = task_queue or DatabaseAdminTaskQueue()

    def awaiting_input(self, state: GraphState) -> TransitionResult:
        return TransitionResult(next_node="retention_strategy" if state.goal.strip() else "awaiting_input",
                                status=None if state.goal.strip() else "waiting")

    def retention_strategy(self, state: GraphState) -> TransitionResult:
        candidates = tree_of_thoughts(problem=state.goal, llm=self.llm, depth=2, beam_width=2, search_strategy="bfs")
        best = candidates[0] if candidates else None
        strategy = {"strategy": best.state if best else "no viable strategy found",
                    "score": best.score if best else 0.0,
                    "rationale": best.rationale if best else ""}
        return TransitionResult(next_node="dispute_react", updates={"data": {**state.data, "retention_strategy": strategy}})

    def dispute_react(self, state: GraphState) -> TransitionResult:
        request_id = state.data.get("hitl_request_id")
        if request_id:
            return self._apply_admin_decision(state, request_id)
        if state.data.get("react_done"):
            return self._finalize(state, state.data.get("customer_response"))

        result = constrained_react(
            task=state.goal, llm=self.llm,
            tools={"check_return_dispute": check_return_dispute,
                   "propose_retention_offer": propose_retention_offer,
                   "escalate_to_hitl": escalate_to_hitl},
            system_prompt=("Handle an escalated, already-closed return dispute. You may inspect it and propose "
                           "an offer, but never finalize it. Escalate on a legal/chargeback threat or uncertainty."),
        )
        summary = {"success": result.success, "output": result.output,
                   "confidence": result.confidence, "iterations": result.iterations}
        if result.escalated or dispute_requires_compliance_review(
            confidence=result.confidence,
            retention_offer_value=float(state.data.get("retention_offer_value", 0) or 0),
            legal_threat=bool(state.data.get("legal_threat", False)),
            policy_violation=bool(state.data.get("policy_violation", False)),
        ):
            return self._pause_for_hitl(state, summary)
        return TransitionResult(next_node="awaiting_customer_response", status="waiting",
                                updates={"data": {**state.data, "react": summary, "react_done": True}})

    def _pause_for_hitl(self, state: GraphState, summary: dict[str, Any]) -> TransitionResult:
        paused_state = state.model_copy(update={"data": {**state.data, "react": summary}})
        pause = self.hitl.pause(paused_state, reason="Dispute requires compliance review before an offer can be finalized.")
        request_id = pause.updates["waiting_request_id"]
        data = {**paused_state.data, "hitl_request_id": request_id}
        queued_state = paused_state.model_copy(update={"data": data, "status": "waiting", "waiting_request_id": request_id})
        self.task_queue.enqueue_hitl(request_id=request_id, run_id=state.run_id, graph_name=state.graph_name,
                                     state=queued_state.model_dump(mode="json"))
        return TransitionResult(next_node="dispute_react", status="waiting",
                                updates={"data": data, "waiting_request_id": request_id})

    def awaiting_customer_response(self, state: GraphState) -> TransitionResult:
        if state.data.get("customer_response") is None:
            return TransitionResult(next_node="awaiting_customer_response", status="waiting")
        return TransitionResult(next_node="dispute_react")

    def _finalize(self, state: GraphState, response: str | None) -> TransitionResult:
        decision = "Approved" if response == "accept" else "Rejected"
        synced = sync_dispute_resolution(return_id=int(state.data["return_id"]), decision=decision,
                                         resolution_note=state.data.get("react", {}).get("output", ""),
                                         simulate_audit_failure=bool(state.data.get("simulate_audit_failure", False)))
        return TransitionResult(next_node="done", status="done", updates={"outputs": {"resolution": synced}})

    def _apply_admin_decision(self, state: GraphState, request_id: str) -> TransitionResult:
        request = self.hitl.get_request(request_id)
        if request is None or request["decision"] is None:
            return TransitionResult(next_node="dispute_react", status="waiting", updates={"waiting_request_id": request_id})
        decision = request["decision"]
        synced = sync_dispute_resolution(return_id=int(state.data["return_id"]),
                                         decision="Approved" if decision == "approve" else "Rejected",
                                         resolution_note=f"Compliance decision: {decision}",
                                         simulate_audit_failure=bool(state.data.get("simulate_audit_failure", False)))
        return TransitionResult(next_node="done", status="done",
                                updates={"data": {**state.data, "hitl_decision": decision},
                                         "outputs": {"resolution": synced, "hitl_decision": decision}})

    def nodes(self) -> dict[str, Any]:
        return {"awaiting_input": self.awaiting_input, "retention_strategy": self.retention_strategy,
                "dispute_react": self.dispute_react, "awaiting_customer_response": self.awaiting_customer_response}
