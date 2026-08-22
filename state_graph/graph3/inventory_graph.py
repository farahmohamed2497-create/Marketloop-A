"""Inventory discrepancy workflow with constrained ReAct and HITL."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from planning_lab.algorithms.react import constrained_react
from state_graph.core.models import GraphState, TransitionResult
from state_graph.hitl.node import HITLNode
from state_graph.hitl.policy import inventory_requires_human_intervention

from .task_queue import AdminTaskQueue, DatabaseAdminTaskQueue


def verify_warehouse_count(*, product_id: int) -> dict[str, Any]:
    return {"product_id": product_id, "verified": True}


def propose_inventory_adjustment(*, product_id: int, quantity_change: int) -> dict[str, Any]:
    return {"product_id": product_id, "quantity_change": quantity_change, "proposed": True}


def escalate_to_hitl(*, reason: str) -> dict[str, Any]:
    return {"escalation_requested": True, "reason": reason}


class InventoryGraph:
    """Never applies a stock adjustment without an explicit admin decision."""

    def __init__(self, *, llm: BaseChatModel, task_queue: AdminTaskQueue | None = None) -> None:
        self.llm = llm
        self.hitl = HITLNode()
        self.task_queue = task_queue or DatabaseAdminTaskQueue()

    def awaiting_input(self, state: GraphState) -> TransitionResult:
        return TransitionResult(
            next_node="inventory_react" if state.goal.strip() else "awaiting_input",
            status=None if state.goal.strip() else "waiting",
        )

    def inventory_react(self, state: GraphState) -> TransitionResult:
        pending_request_id = state.data.get("hitl_request_id")
        if pending_request_id:
            return self._apply_admin_decision(state, pending_request_id)

        result = constrained_react(
            task=state.goal,
            llm=self.llm,
            tools={
                "verify_warehouse_count": verify_warehouse_count,
                "propose_inventory_adjustment": propose_inventory_adjustment,
                "escalate_to_hitl": escalate_to_hitl,
            },
            system_prompt=(
                "You investigate inventory discrepancies. You may verify counts and "
                "propose an adjustment, but you must never apply an adjustment. "
                "Escalate when uncertain."
            ),
        )
        variance = int(state.data.get("quantity_variance", 0))
        policy_violation = bool(state.data.get("policy_violation", False))
        summary = {
            "success": result.success,
            "output": result.output,
            "confidence": result.confidence,
            "iterations": result.iterations,
        }
        if result.escalated or inventory_requires_human_intervention(
            confidence=result.confidence,
            quantity_variance=variance,
            policy_violation=policy_violation,
        ):
            paused_state = state.model_copy(update={"data": {**state.data, "react": summary}})
            pause = self.hitl.pause(
                paused_state,
                reason="Inventory change requires warehouse-admin approval.",
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
                next_node="inventory_react",
                status="waiting",
                updates={"data": persisted_data, "waiting_request_id": request_id},
            )

        return TransitionResult(
            next_node="done",
            status="done",
            updates={"data": {**state.data, "react": summary}, "outputs": {"resolution": result.output}},
        )

    def _apply_admin_decision(self, state: GraphState, request_id: str) -> TransitionResult:
        request = self.hitl.get_request(request_id)
        if request is None or request["decision"] is None:
            return TransitionResult(next_node="inventory_react", status="waiting", updates={"waiting_request_id": request_id})
        decision = request["decision"]
        return TransitionResult(
            next_node="done",
            status="done",
            updates={
                "data": {**state.data, "hitl_decision": decision},
                "outputs": {"resolution": "Inventory adjustment approved." if decision == "approve" else "Inventory adjustment rejected.", "hitl_decision": decision},
            },
        )

    def nodes(self) -> dict[str, Any]:
        return {"awaiting_input": self.awaiting_input, "inventory_react": self.inventory_react}
