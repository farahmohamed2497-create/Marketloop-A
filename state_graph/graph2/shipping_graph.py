from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from planning_lab.algorithms.react import ReactResult, constrained_react
from state_graph.core.models import GraphState, TransitionResult
from state_graph.hitl.node import HITLNode
from state_graph.hitl.policy import shipping_requires_human_intervention

from .tools import check_tracking, escalate_to_hitl, open_carrier_claim

SHIPPING_TOOLS = {
    "check_tracking": check_tracking,
    "open_carrier_claim": open_carrier_claim,
    "escalate_to_hitl": escalate_to_hitl,
}


class ShippingGraph:
    """
    State graph for shipping / delivery issue resolution.

    Addition 1 (task decomposition) breaks the customer's shipping
    issue into concrete subtasks. Addition 2 (constrained ReAct) then
    works through those subtasks using ONLY the tools in
    SHIPPING_TOOLS. When the ReAct run's confidence is too low, the
    claim amount is too high, or the proposed action contradicts
    policy, the graph pauses through the persistent HITL node instead
    of letting the agent decide alone.
    """

    def __init__(self, *, llm: BaseChatModel) -> None:
        self.llm = llm
        self.hitl = HITLNode()

    # ------------------------------------------------------------------
    # awaiting_input
    # ------------------------------------------------------------------

    def awaiting_input(self, state: GraphState) -> TransitionResult:
        """Wait until a shipping issue is available."""
        if not state.goal.strip():
            return TransitionResult(
                next_node="awaiting_input",
                status="waiting",
            )

        return TransitionResult(next_node="decompose")

    # ------------------------------------------------------------------
    # Addition 1: task decomposition
    # ------------------------------------------------------------------

    def decompose(self, state: GraphState) -> TransitionResult:
        """
        Break the shipping issue into concrete subtasks.

        Kept intentionally simple — this graph's assignment is the
        HITL escalation work; decomposition just needs to hand the
        constrained-ReAct node a clear, ordered set of steps instead
        of the customer's raw free text.
        """
        prompt = (
            "Break the following shipping issue into a short ordered "
            "list of concrete subtasks an agent could act on, using "
            "only tracking checks and carrier claims. "
            "Return one subtask per line, no numbering.\n\n"
            f"Issue: {state.goal}"
        )

        response = self.llm.invoke(prompt)

        subtasks = [
            line.strip("- ").strip()
            for line in response.content.splitlines()
            if line.strip()
        ]

        return TransitionResult(
            next_node="constrained_react",
            updates={
                "data": {
                    **state.data,
                    "subtasks": subtasks,
                },
            },
        )

    # ------------------------------------------------------------------
    # Addition 2: constrained ReAct + HITL
    # ------------------------------------------------------------------

    def constrained_react_node(self, state: GraphState) -> TransitionResult:
        """
        Resolve the subtasks with a constrained ReAct loop, or — if
        this is a resumed run — pick up the admin's HITL decision
        instead of re-running the loop from scratch.
        """
        pending_request_id = state.data.get("hitl_request_id")

        if pending_request_id is not None:
            return self._resume_from_hitl_decision(state, pending_request_id)

        return self._run_react_and_maybe_pause(state)

    def _run_react_and_maybe_pause(self, state: GraphState) -> TransitionResult:
        subtasks = state.data.get("subtasks", [])
        task = state.goal + "\n\nSubtasks:\n" + "\n".join(subtasks)

        result = constrained_react(
            task=task,
            llm=self.llm,
            tools=SHIPPING_TOOLS,
        )

        claim_amount = self._extract_claim_amount(result)
        policy_violation = self._detect_policy_violation(result)

        updated_data = {
            **state.data,
            "react": {
                "success": result.success,
                "output": result.output,
                "confidence": result.confidence,
                "iterations": result.iterations,
                "tool_calls": [
                    {
                        "tool_name": call.tool_name,
                        "arguments": call.arguments,
                        "result": call.result,
                    }
                    for call in result.tool_calls
                ],
            },
        }

        if shipping_requires_human_intervention(
            confidence=result.confidence,
            claim_amount=claim_amount,
            policy_violation=policy_violation,
        ):
            return self._pause_for_review(
                state,
                updated_data=updated_data,
                confidence=result.confidence,
                claim_amount=claim_amount,
                policy_violation=policy_violation,
            )

        return TransitionResult(
            next_node="done",
            status="done",
            updates={
                "data": updated_data,
                "outputs": {"resolution": result.output},
            },
        )

    def _pause_for_review(
        self,
        state: GraphState,
        *,
        updated_data: dict[str, Any],
        confidence: float,
        claim_amount: float | None,
        policy_violation: bool,
    ) -> TransitionResult:
        """
        Pause via HITLNode, and additionally stash the resulting
        request id into `data` (not just the transient
        `waiting_request_id` field) so that a resumed run can still
        look up and apply the admin's decision even after
        StateGraphEngine.resume() clears `waiting_request_id`.
        """
        paused_state = state.model_copy(update={"data": updated_data})

        pause_result = self.hitl.pause(
            paused_state,
            reason=(
                "Shipping resolution requires human review because "
                "the configured HITL policy was triggered "
                f"(confidence={confidence:.2f}, "
                f"claim_amount={claim_amount}, "
                f"policy_violation={policy_violation})."
            ),
        )

        request_id = pause_result.updates.get("waiting_request_id")

        merged_updates = dict(pause_result.updates)
        merged_updates["data"] = {
            **updated_data,
            "hitl_request_id": request_id,
        }

        return TransitionResult(
            next_node=pause_result.next_node,
            status=pause_result.status,
            updates=merged_updates,
            error=pause_result.error,
        )

    def _resume_from_hitl_decision(
        self,
        state: GraphState,
        request_id: str,
    ) -> TransitionResult:
        """
        Apply the admin's decision from HITL_Requests instead of
        proceeding as if nothing happened.

        Defensive by design: if StateGraphEngine.resume() was called
        before the admin actually recorded a decision, this re-pauses
        rather than silently finishing the run.
        """
        request = self.hitl.get_request(request_id)

        if request is None or request.get("decision") is None:
            # Don't open a second HITL_Requests row for the same
            # wait — just re-assert the waiting status against the
            # SAME request_id, since it's still genuinely pending.
            return TransitionResult(
                next_node=state.current_node,
                status="waiting",
                updates={"waiting_request_id": request_id},
            )

        decision = request["decision"]
        react_summary = state.data.get("react", {})

        updated_data = {
            key: value
            for key, value in state.data.items()
            if key != "hitl_request_id"
        }
        updated_data["hitl_decision"] = decision

        if decision == "approve":
            resolution = react_summary.get("output", "")
        else:
            resolution = "Claim rejected by human reviewer."

        return TransitionResult(
            next_node="done",
            status="done",
            updates={
                "data": updated_data,
                "outputs": {
                    "resolution": resolution,
                    "hitl_decision": decision,
                },
            },
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_claim_amount(result: ReactResult) -> float | None:
        """Pull a claim amount out of any open_carrier_claim tool call."""
        for call in result.tool_calls:
            if call.tool_name == "open_carrier_claim":
                return call.arguments.get("claim_amount")
        return None

    @staticmethod
    def _detect_policy_violation(result: ReactResult) -> bool:
        """
        A policy violation here means the agent opened a carrier claim
        for a shipment that check_tracking reported as already
        delivered — filing a loss/damage claim against a confirmed
        delivery contradicts stated policy and must not be decided by
        the agent alone.
        """
        delivered = any(
            call.tool_name == "check_tracking"
            and isinstance(call.result, dict)
            and call.result.get("status") == "delivered"
            for call in result.tool_calls
        )
        claimed = any(call.tool_name == "open_carrier_claim" for call in result.tool_calls)
        return delivered and claimed

    def nodes(self) -> dict[str, Any]:
        return {
            "awaiting_input": self.awaiting_input,
            "decompose": self.decompose,
            "constrained_react": self.constrained_react_node,
        }
