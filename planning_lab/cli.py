from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI


from .algorithms import (
    decompose_goal,
    dynamic_decomposition,
    execute_plan,
    final_output,
    flatten_lats_tree,
    lats,
    plan_and_solve,
    reflexion,
    reflect_and_refine,
    Environment,
    tree_of_thoughts,
)
from .evaluation import (
    evaluate_decomposition,
    evaluate_plan_and_solve,
    evaluate_self_refine,
)
from .mcp_executor import MarketLoopMCPExecutor
from .sales_audit_environment import SalesAuditEnvironment



ROOT = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description="Week 4: decomposition, planning, and reflection lab")
    cli.add_argument("goal", nargs="?", default="Design a 60-minute phishing-awareness workshop for new employees")
    cli.add_argument(
        "--mode",
        choices=["dag", "dynamic", "ps", "tot", "reflexion", "lats"],
        default="dag",
    )
    cli.add_argument("--model", default="mistral-small-latest")
    cli.add_argument("--depth", type=int, default=2, choices=range(1, 4))
    cli.add_argument("--beam-width", type=int, default=2, choices=range(1, 4))
    cli.add_argument("--search-strategy", choices=["bfs", "dfs"], default="bfs")
    cli.add_argument("--prune-threshold", type=float, default=0.0)
    cli.add_argument("--max-trials", type=int, default=3, choices=range(1, 6))
    cli.add_argument("--memory-size", type=int, default=3, choices=range(1, 6))
    cli.add_argument("--iterations", type=int, default=2, choices=range(1, 6))
    cli.add_argument("--n-actions", type=int, default=2, choices=range(1, 4))
    cli.add_argument("--success-threshold", type=float, default=0.6)
    cli.add_argument("--grounded-sales-audit", action="store_true")
    cli.add_argument("--start-date", default="2026-01-01")
    cli.add_argument("--end-date", default="2026-01-31")
    cli.add_argument("--no-reflection", action="store_true")
    return cli


def save_artifact(payload: dict) -> Path:
    artifact_dir = ROOT / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = artifact_dir / f"run-{stamp}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> None:
    # Mistral may return arrows, em dashes, or other characters that Windows'
    # legacy cp1252 console cannot encode. UTF-8 keeps CLI output portable.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parser().parse_args()
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is missing; add it to .env")
    llm = ChatMistralAI(
        api_key=api_key,
        model=args.model,
        random_seed=42,
        max_retries=2,
    )
    payload: dict = {"mode": args.mode, "model": args.model, "goal": args.goal}

    if args.mode == "dag":
        plan = decompose_goal(args.goal, llm)
        print("Execution batches:", plan.execution_batches())

        decomposition_evaluation = evaluate_decomposition(plan)

        outputs = execute_plan(plan, llm)
        draft = final_output(plan, outputs)

        reflection = (
            reflect_and_refine(args.goal, draft, llm)
            if not args.no_reflection
            else None
        )

        result = reflection.revised if reflection else draft

        payload.update(
            plan=plan.model_dump(),
            outputs=outputs,
            result=result,
            evaluation={
                "decomposition": {
                    "metrics": {
                        "total": decomposition_evaluation["metrics"].total,
                        "passed": decomposition_evaluation["metrics"].passed,
                        "failed": decomposition_evaluation["metrics"].failed,
                        "pass_rate": decomposition_evaluation["metrics"].pass_rate,
                    },
                    "execution_batches": decomposition_evaluation[
                        "execution_batches"
                    ],
                    "terminal_tasks": decomposition_evaluation[
                        "terminal_tasks"
                    ],
                }
            },
        )

        if reflection:
            self_refine_evaluation = evaluate_self_refine(
                draft=reflection.draft,
                revised=reflection.revised,
                grounded_issues=reflection.grounded_issues,
            )

            payload["reflection"] = {
                "grounded_issues": reflection.grounded_issues,
                "critique": reflection.critique,
                "revised": reflection.revised != reflection.draft,
            }

            payload["evaluation"]["self_refine"] = {
                "metrics": {
                    "total": self_refine_evaluation["metrics"].total,
                    "passed": self_refine_evaluation["metrics"].passed,
                    "failed": self_refine_evaluation["metrics"].failed,
                    "pass_rate": self_refine_evaluation["metrics"].pass_rate,
                },
                "draft_changed": self_refine_evaluation["draft_changed"],
                "grounded_issue_count": self_refine_evaluation[
                    "grounded_issue_count"
                ],
            }
    elif args.mode == "dynamic":
        history = dynamic_decomposition(args.goal, llm)
        result = history[-1][1] if history else "Planner reported the goal was already complete."
        payload.update(history=history, result=result)
    elif args.mode == "ps":
        result = plan_and_solve(args.goal, llm)

        ps_evaluation = evaluate_plan_and_solve(result)

        payload.update(
            result=result,
            evaluation={
                "plan_and_solve": {
                    "metrics": {
                        "total": ps_evaluation["metrics"].total,
                        "passed": ps_evaluation["metrics"].passed,
                        "failed": ps_evaluation["metrics"].failed,
                        "pass_rate": ps_evaluation["metrics"].pass_rate,
                    },
                }
            },
        )
    elif args.mode == "tot":
        thoughts = tree_of_thoughts(
            args.goal,
            llm,
            args.depth,
            args.beam_width,
            args.search_strategy,
            args.prune_threshold,
        )
        result = thoughts[0].state if thoughts else "No viable thought survived."
        payload.update(
            search_strategy=args.search_strategy,
            prune_threshold=args.prune_threshold,
            thoughts=[thought.model_dump() for thought in thoughts],
            result=result,
        )
    elif args.mode == "reflexion":
        environment = (
            SalesAuditEnvironment(MarketLoopMCPExecutor(), args.start_date, args.end_date)
            if args.grounded_sales_audit
            else Environment(success_threshold=args.success_threshold)
        )
        outcome = reflexion(args.goal, llm, environment, args.max_trials, args.memory_size)
        result = outcome.output
        payload.update(
            success=outcome.success,
            trials=[
                {
                    "number": trial.number,
                    "attempt": trial.attempt,
                    "feedback": trial.feedback.model_dump(),
                    "reflection": trial.reflection,
                }
                for trial in outcome.trials
            ],
            memory=outcome.memory,
            result=result,
        )
    else:
        environment = (
            SalesAuditEnvironment(MarketLoopMCPExecutor(), args.start_date, args.end_date)
            if args.grounded_sales_audit
            else Environment(success_threshold=args.success_threshold)
        )
        outcome = lats(args.goal, llm, environment, args.iterations, args.n_actions)
        result = outcome.output
        payload.update(
            success=outcome.success,
            best_score=outcome.best_score,
            iterations=outcome.iterations,
            tree=flatten_lats_tree(outcome.root),
            result=result,
        )

    artifact = save_artifact(payload)
    print("\nRESULT\n======\n" + result)
    print(f"\nRun artifact: {artifact}")


if __name__ == "__main__":
    main()