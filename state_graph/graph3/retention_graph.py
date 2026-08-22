"""Graph 3 — Retention: transition functions.

Ticket items:
  feat(graph3): implement transition functions between states
  feat(graph3): implement [LLM addition #1] node   -> Constrained ReAct

LLM addition #2 (RAG) is teammate's — see `policy_lookup` below for the
seam it plugs into. Nothing here depends on RAG actually being
implemented yet; `policy_lookup` is a pass-through stub today.
"""
from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from planning_lab.algorithms.react import constrained_react
from state_graph.core.models import GraphState, TransitionResult
from state_graph.hitl.node import HITLNode
from state_graph.hitl.policy import dispute_requires_compliance_review

from .state import RetentionNode
from .tools import apply_discount_code, escalate_to_legal, send_retention_offer

MAX_NEGOTIATION_ROUNDS = 3


class RetentionGraph:
    """Never applies a discount or lets an offer stand without either a
    real customer accept or an explicit compliance-admin decision."""

    def __init__(self, *, llm: BaseChatModel) -> None:
        self.llm = llm
        self.hitl = HITLNode()

    # ---- awaiting_input -------------------------------------------- #

    def awaiting_input(self, state: GraphState) -> TransitionResult:
        if not state.goal.strip():
            return TransitionResult(next_node=RetentionNode.AWAITING_INPUT.value, status="waiting")
        return TransitionResult(next_node=RetentionNode.POLICY_LOOKUP.value)

    # ---- policy_lookup (RAG seam — teammate implements addition #2) - #

    def policy_lookup(self, state: GraphState) -> TransitionResult:
        """Pass-through stub. Teammate replaces this body with the real
        RAG retrieval (company retention-offer policy + legal-escalation
        criteria) and writes the result into `data["policy_context"]`.
        The edge (policy_lookup -> retention_react) is already wired so
        landing that change doesn't require touching graph.py's
        TransitionTable at all — just this function body.
        """
        return TransitionResult(
            next_node=RetentionNode.RETENTION_REACT.value,
            updates={"data": {**state.data, "policy_context": state.data.get("policy_context", "")}},
        )

    # ---- retention_react (Constrained ReAct — LLM addition #1) ------ #

    def retention_react(self, state: GraphState) -> TransitionResult:
        pending_request_id = state.data.get("hitl_request_id")
        if pending_request_id:
            return self._apply_admin_decision(state, pending_request_id)

        result = constrained_react(
            task=state.goal,
            llm=self.llm,
            tools={
                "send_retention_offer": send_retention_offer,
                "escalate_to_legal": escalate_to_legal,
                "apply_discount_code": apply_discount_code,
            },
            system_prompt=(
                "You handle a subscription cancellation. Use the retention policy "
                "context provided to decide between proposing a retention offer, "
                "applying an approved discount directly, or escalating to Legal. "
                "You may never finalize a discount above policy caps yourself, and "
                "you must escalate on any legal or chargeback threat rather than "
                "attempt to resolve it with a discount."
            ),
        )

        offer_value = float(state.data.get("retention_offer_value", 0) or 0)
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
            retention_offer_value=offer_value,
            legal_threat=legal_threat,
            policy_violation=policy_violation,
        ):
            return self._pause_for_hitl(state, summary)

        chosen_action = state.data.get("proposed_action", "send_retention_offer")

        if chosen_action == "apply_discount_code":
            # Within policy caps and not flagged above -> apply directly,
            # no customer wait needed for an already-granted discount.
            # This must actually call the tool -- an earlier draft only
            # marked the run "done" without writing anything, which is
            # exactly the kind of gap a shallow status-only test won't
            # catch (see test_retention_react_direct_discount_within_policy_goes_done,
            # which now asserts the write happened, not just the status).
            sync_result = apply_discount_code(
                subscription_id=int(state.data["subscription_id"]),
                discount_pct=float(state.data.get("proposed_discount_pct", 0.1)),
                resolution_note="Direct discount within policy caps (no HITL required).",
                simulate_audit_failure=bool(state.data.get("simulate_audit_failure", False)),
            )
            return TransitionResult(
                next_node=RetentionNode.DONE.value,
                status="done",
                updates={
                    "data": {**state.data, "react": summary},
                    "outputs": {"resolution": sync_result, "react": summary},
                },
            )

        # Default path: an offer was proposed and now needs the
        # customer's answer -- a real external wait, not HITL.
        round_num = int(state.data.get("negotiation_round", 0)) + 1
        return TransitionResult(
            next_node=RetentionNode.AWAITING_CUSTOMER_RESPONSE.value,
            status="waiting",
            updates={
                "data": {
                    **state.data,
                    "react": summary,
                    "negotiation_round": round_num,
                    "customer_response": None,
                },
            },
        )

    # ---- awaiting_customer_response (real external wait / cycle) --- #

    def awaiting_customer_response(self, state: GraphState) -> TransitionResult:
        response = state.data.get("customer_response")

        if response is None:
            return TransitionResult(
                next_node=RetentionNode.AWAITING_CUSTOMER_RESPONSE.value, status="waiting"
            )

        if response == "accept":
            return TransitionResult(
                next_node=RetentionNode.DONE.value,
                status="done",
                updates={"outputs": {"resolution": "retained", "round": state.data.get("negotiation_round")}},
            )

        # Rejected. If we still have negotiation rounds left, cycle back
        # into retention_react for another (harsher) offer or an
        # escalation decision. This loop-back is the genuine cycle: the
        # SAME state (retention_react) is revisited with new information
        # (the rejection), which a DAG cannot express.
        round_num = int(state.data.get("negotiation_round", 0))
        if round_num >= MAX_NEGOTIATION_ROUNDS:
            return TransitionResult(
                next_node=RetentionNode.RETENTION_REACT.value,
                updates={"data": {**state.data, "policy_violation": True,
                                   "hitl_reason_hint": "negotiation rounds exhausted"}},
            )

        return TransitionResult(
            next_node=RetentionNode.RETENTION_REACT.value,
            updates={"data": {**state.data, "customer_response": None}},
        )

    # ---- HITL pause / resume ----------------------------------------- #

    def _pause_for_hitl(self, state: GraphState, summary: dict[str, Any]) -> TransitionResult:
        paused_state = state.model_copy(update={"data": {**state.data, "react": summary}})
        pause = self.hitl.pause(
            paused_state,
            reason="Retention offer requires compliance/admin review before it can be finalized.",
        )
        request_id = pause.updates["waiting_request_id"]
        persisted_data = {**paused_state.data, "hitl_request_id": request_id}
        return TransitionResult(
            next_node=RetentionNode.RETENTION_REACT.value,
            status="waiting",
            updates={"data": persisted_data, "waiting_request_id": request_id},
        )

    def _apply_admin_decision(self, state: GraphState, request_id: str) -> TransitionResult:
        request = self.hitl.get_request(request_id)
        if request is None or request["decision"] is None:
            return TransitionResult(
                next_node=RetentionNode.RETENTION_REACT.value,
                status="waiting",
                updates={"waiting_request_id": request_id},
            )

        decision = request["decision"]
        if decision != "approve":
            return TransitionResult(
                next_node=RetentionNode.DONE.value,
                status="done",
                updates={
                    "data": {**state.data, "hitl_decision": decision},
                    "outputs": {"resolution": "churned", "hitl_decision": decision},
                },
            )

        # NOTE: `retention_offer_value` is a DOLLAR figure used for the
        # HITL threshold check above (offer_value_threshold=500.0). It is
        # NOT the same thing as `discount_pct`, which apply_discount_code
        # requires to be a (0, 1] fraction. Conflating the two was a real
        # bug in an earlier draft of this file -- caught during review,
        # not by the test suite, which is why `proposed_discount_pct` is
        # now a separate, explicit field with its own default.
        sync_result = apply_discount_code(
            subscription_id=int(state.data["subscription_id"]),
            discount_pct=float(state.data.get("proposed_discount_pct", 0.1)),
            resolution_note=f"Admin-approved retention offer: {decision}",
            simulate_audit_failure=bool(state.data.get("simulate_audit_failure", False)),
        )
        return TransitionResult(
            next_node=RetentionNode.DONE.value,
            status="done",
            updates={
                "data": {**state.data, "hitl_decision": decision},
                "outputs": {"resolution": sync_result, "hitl_decision": decision},
            },
        )

    def nodes(self) -> dict[str, Any]:
        from .state import build_node_registry
        return build_node_registry(self)