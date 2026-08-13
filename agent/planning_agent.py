"""Separate Sales Audit planning agent; it does not alter the Memory/RAG path."""

from __future__ import annotations

from typing import Any

from planning_lab.algorithms.decomposition import (
    decompose_goal,
    execute_plan,
    final_output,
)
from planning_lab.algorithms.lats import lats
from planning_lab.algorithms.plan_and_solve import plan_and_solve
from planning_lab.algorithms.reflexion import reflexion
from planning_lab.algorithms.tree_of_thoughts import tree_of_thoughts
from planning_lab.mcp_executor import MarketLoopMCPExecutor
from planning_lab.models import Plan
from planning_lab.routing import PlanningMethod, route_sales_audit_subtask
from planning_lab.sales_audit_environment import SalesAuditEnvironment


class SalesAuditDecompositionRun:
    """Evidence returned by the toolkit DAG execution for one audit request."""

    def __init__(self, plan: Plan, outputs: dict[str, str], result: str) -> None:
        self.plan = plan
        self.outputs = outputs
        self.result = result


class SalesAuditPlanningAgent:
    """Route Sales Audit subtasks to the planning strategy that fits them."""

    def __init__(
        self,
        llm: Any,
        executor: MarketLoopMCPExecutor | None = None,
        *,
        start_date: str = "2026-01-01",
        end_date: str = "2026-01-31",
    ) -> None:
        self.llm = llm
        self.executor = executor or MarketLoopMCPExecutor()
        self.start_date = start_date
        self.end_date = end_date

    def route(self, subtask: str) -> PlanningMethod:
        return route_sales_audit_subtask(subtask)

    def run_decomposition_first(self, goal: str) -> SalesAuditDecompositionRun:
        """Run the reference toolkit's DAG against MarketLoop MCP tool nodes."""
        plan = decompose_goal(goal, self.llm)
        outputs = execute_plan(plan, self.llm, task_executor=self.executor)
        return SalesAuditDecompositionRun(plan, outputs, final_output(plan, outputs))

    def solve(self, subtask: str) -> str:
        method = self.route(subtask)

        if method is PlanningMethod.MCP:
            return self.executor.execute(
                "generate_sales_audit_report",
                {"start_date": self.start_date, "end_date": self.end_date},
            )

        if method is PlanningMethod.TREE_OF_THOUGHTS:
            thoughts = tree_of_thoughts(
                subtask,
                self.llm,
                search_strategy="bfs",
                prune_threshold=0.5,
            )
            return thoughts[0].state if thoughts else "No viable action survived pruning."

        if method is PlanningMethod.REFLEXION:
            outcome = reflexion(
                subtask,
                self.llm,
                SalesAuditEnvironment(self.executor, self.start_date, self.end_date),
            )
            return outcome.output

        if method is PlanningMethod.LATS:
            outcome = lats(
                subtask,
                self.llm,
                SalesAuditEnvironment(self.executor, self.start_date, self.end_date),
            )
            return outcome.output

        return plan_and_solve(subtask, self.llm)