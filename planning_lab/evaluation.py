from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationMetrics:
    total: int
    passed: int
    failed: int
    pass_rate: float

    @property
    def success(self) -> bool:
        return self.failed == 0


def calculate_metrics(results: list[bool]) -> EvaluationMetrics:
    """Calculate basic evaluation metrics from boolean test results."""

    total = len(results)
    passed = sum(results)
    failed = total - passed

    pass_rate = passed / total if total else 0.0

    return EvaluationMetrics(
        total=total,
        passed=passed,
        failed=failed,
        pass_rate=pass_rate,
    )


def evaluate_decomposition(plan) -> dict[str, object]:
    """Evaluate structural properties of a generated decomposition."""

    batches = plan.execution_batches()
    terminals = plan.terminal_tasks()

    has_parallelism = any(len(batch) > 1 for batch in batches)
    has_single_terminal = len(terminals) == 1

    checks = [
        bool(plan.tasks),
        has_single_terminal,
        bool(batches),
    ]

    metrics = calculate_metrics(checks)

    return {
        "component": "decomposition",
        "metrics": metrics,
        "execution_batches": batches,
        "terminal_tasks": terminals,
    }


def evaluate_plan_and_solve(output: str) -> dict[str, object]:
    """Evaluate whether Plan-and-Solve output contains both phases."""

    normalized = output.lower()

    has_plan = "plan" in normalized
    has_solution = "solution" in normalized

    checks = [
        bool(output.strip()),
        has_plan,
        has_solution,
    ]

    metrics = calculate_metrics(checks)

    return {
        "component": "plan_and_solve",
        "metrics": metrics,
    }


def evaluate_self_refine(
    draft: str,
    revised: str,
    grounded_issues: list[str],
) -> dict[str, object]:
    """Evaluate whether Self-Refine produced a meaningful refinement."""

    draft_non_empty = bool(draft.strip())
    revised_non_empty = bool(revised.strip())

    changed_when_needed = (
        bool(grounded_issues) and revised.strip() != draft.strip()
    )

    no_unnecessary_change = (
        not grounded_issues and revised.strip() == draft.strip()
    )

    checks = [
        draft_non_empty,
        revised_non_empty,
        changed_when_needed or no_unnecessary_change,
    ]

    metrics = calculate_metrics(checks)

    return {
        "component": "self_refine",
        "metrics": metrics,
        "draft_changed": revised.strip() != draft.strip(),
        "grounded_issue_count": len(grounded_issues),
    }