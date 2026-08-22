from __future__ import annotations


from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from planning_lab.algorithms.react import ReactResult, constrained_react
from state_graph.core.models import GraphState, TransitionResult
from state_graph.hitl.node import HITLNode
from state_graph.hitl.policy import shipping_requires_human_intervention
from state_graph.checkpointing.store import CheckpointStore
from state_graph.tickets.service import FailureTicketService
from .tools import check_tracking, escalate_to_hitl, open_carrier_claim
from .decomposition import ShippingTaskDecomposer


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

    def __init__(
            self,
            *,
            llm: BaseChatModel,
            checkpoint_store: CheckpointStore | None = None,
            failure_ticket_service: FailureTicketService | None = None,
    ) -> None:
        self.llm = llm
        self.decomposer = ShippingTaskDecomposer(llm)
        self.hitl = HITLNode()

        self.checkpoints = checkpoint_store or CheckpointStore()
        self.failure_tickets = (
                failure_ticket_service or FailureTicketService()
        )

    # ------------------------------------------------------------------
    # checkpoint helper
    # ------------------------------------------------------------------
    def _checkpoint_transition(
            self,
            state: GraphState,
            result: TransitionResult,
    ) -> TransitionResult:
        """
        Persist the state produced by a meaningful graph transition.

        The checkpoint represents the state that should be resumed from
        if the process dies after this transition.
        """
        updates = result.updates or {}

        checkpoint_data = updates.get("data", state.data)
        checkpoint_outputs = updates.get("outputs", state.outputs)

        checkpoint_updates: dict[str, Any] = {
            "current_node": result.next_node,
            "status": result.status or state.status,
            "data": checkpoint_data,
            "outputs": checkpoint_outputs,
            "transition_count": state.transition_count + 1,
        }

        if "waiting_request_id" in updates:
            checkpoint_updates["waiting_request_id"] = updates[
                "waiting_request_id"
            ]

        checkpoint_state = state.model_copy(
            update=checkpoint_updates
        )

        self.checkpoints.save(checkpoint_state)

        return result

    # ------------------------------------------------------------------
    # awaiting_input
    # ------------------------------------------------------------------

    def awaiting_input(self, state: GraphState) -> TransitionResult:
        """Wait until a shipping issue is available."""

        if not state.goal.strip():
            return self._checkpoint_transition(
                state,
                TransitionResult(
                    next_node="awaiting_input",
                    status="waiting",
                ),
            )

        return self._checkpoint_transition(
            state,
            TransitionResult(
                next_node="decompose",
            ),
        )

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
        subtasks = self.decomposer.decompose(state.goal)

        return self._checkpoint_transition(
            state,
            TransitionResult(
                next_node="constrained_react",
                updates={
                    "data": {
                        **state.data,
                        "subtasks": subtasks,
                    },
                },
            ),
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

    def awaiting_carrier(self, state: GraphState) -> TransitionResult:
        """Pause until the external carrier supplies the requested evidence."""

        if not state.data.get("carrier_response"):
            return TransitionResult(
                next_node="awaiting_carrier",
                status="waiting",
            )

        return TransitionResult(next_node="constrained_react")

    def _run_react_and_maybe_pause(self, state: GraphState) -> TransitionResult:
        subtasks = state.data.get("subtasks", [])
        task = state.goal + "\n\nSubtasks:\n" + "\n".join(subtasks)

        try:
            result = constrained_react(
                task=task,
                llm=self.llm,
                tools=SHIPPING_TOOLS,
            )

            self._validate_react_result(result)

        except (Exception,) as exc:
            return self._create_failure_ticket(
                state,
                exc,
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

        next_node = (
            "awaiting_carrier"
            if state.data.get("carrier_response_required")
            else "done"
        )

        return self._checkpoint_transition(
            state,
            TransitionResult(
                next_node=next_node,
                status="waiting" if next_node == "awaiting_carrier" else "done",
                updates={
                    "data": updated_data,
                    "outputs": {
                        "resolution": result.output,
                    },
                },
            ),
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

        return self._checkpoint_transition(
            state,
            TransitionResult(
                next_node=pause_result.next_node,
                status=pause_result.status,
                updates=merged_updates,
                error=pause_result.error,
            ),
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
    # Failure Detection
    # ------------------------------------------------------------------

    def _validate_react_result(self, result: ReactResult) -> None:
        """
        Detect failures that cannot be fixed by simply retrying.

        Graph 2 treats tool errors, malformed tool responses,
        schema violations, and contradictory tracking data as
        unplanned execution failures.
        """

        escalation_requested = any(
            call.tool_name == "escalate_to_hitl"
            for call in result.tool_calls
        )

        # A constrained ReAct run deliberately stops with success=False
        # after choosing the whitelisted escalation tool. That is an
        # expected HITL branch, not an unplanned tool failure.
        if not result.success and not escalation_requested:
            raise RuntimeError(
                "Constrained ReAct failed to produce a successful result."
            )

        statuses_by_tracking: dict[str, set[str]] = {}

        for call in result.tool_calls:
            tool_result = call.result

            if isinstance(tool_result, Exception):
                raise RuntimeError(
                    f"Tool {call.tool_name} failed: {tool_result}"
                )

            if isinstance(tool_result, dict) and "error" in tool_result:
                raise RuntimeError(
                    f"Tool {call.tool_name} returned an error: "
                    f"{tool_result['error']}"
                )

            if call.tool_name == "check_tracking":
                if not isinstance(tool_result, dict):
                    raise ValueError(
                        "check_tracking returned an invalid response schema."
                    )

                required_fields = {
                    "tracking_number",
                    "status",
                }

                if not required_fields.issubset(tool_result):
                    raise ValueError(
                        "check_tracking response is missing required fields."
                    )

                tracking_number = str(
                    tool_result["tracking_number"]
                )

                status = str(
                    tool_result["status"]
                ).strip().lower()

                allowed_statuses = {
                    "in_transit",
                    "delivered",
                    "lost",
                    "investigating",
                }

                if status not in allowed_statuses:
                    raise ValueError(
                        f"Unknown tracking status: {status!r}"
                    )

                statuses_by_tracking.setdefault(
                    tracking_number,
                    set(),
                ).add(status)

            elif call.tool_name == "open_carrier_claim":
                if not isinstance(tool_result, dict):
                    raise ValueError(
                        "open_carrier_claim returned an invalid response schema."
                    )

                if "claim_id" not in tool_result:
                    raise ValueError(
                        "open_carrier_claim response is missing claim_id."
                    )

                if "status" not in tool_result:
                    raise ValueError(
                        "open_carrier_claim response is missing status."
                    )

            elif call.tool_name == "escalate_to_hitl":
                if not isinstance(tool_result, dict):
                    raise ValueError(
                        "escalate_to_hitl returned an invalid response schema."
                    )

        contradictory_tracking = {
            tracking_number: statuses
            for tracking_number, statuses in statuses_by_tracking.items()
            if len(statuses) > 1
        }

        if contradictory_tracking:
            raise RuntimeError(
                "Contradictory carrier tracking data detected: "
                f"{contradictory_tracking}"
            )

    def _create_failure_ticket(
            self,
            state: GraphState,
            error: Exception,
    ) -> TransitionResult:
        """Create a persistent ticket for an unplanned execution failure."""

        ticket_id = self.failure_tickets.create_ticket(
            run_id=state.run_id,
            graph_name=state.graph_name,
            node_name=state.current_node,
            error=str(error),
            state=state.model_dump(mode="json"),
        )

        return TransitionResult(
            next_node=state.current_node,
            status="failed",
            updates={
                "waiting_ticket_id": ticket_id,
                "data": {
                    **state.data,
                    "failure_ticket_id": ticket_id,
                },
            },
            error=str(error),
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
            "awaiting_carrier": self.awaiting_carrier,
        }
