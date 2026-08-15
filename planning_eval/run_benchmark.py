"""
planning_eval/run_benchmark.py

Real benchmark runner for the MarketLoop Sales Audit planning agent.

No mock database.
No synthetic outcomes.
No random success/failure.
No fake latency.
No fake token counts.

The benchmark runs the real planning implementations against the real
MarketLoop SQLite/MCP-backed environment and writes one JSON trace per run.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import RateLimitError
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq

from planning_eval.test_cases import TEST_CASES, TestCase
from planning_lab.mcp_executor import MarketLoopMCPExecutor
from planning_lab.algorithms.decomposition import (
    decompose_goal,
    execute_plan,
    final_output,
)
from planning_lab.algorithms.dynamic_decomposition import (
    dynamic_decomposition,
)
from planning_eval.grounded_environment import CaseGroundedEnvironment as Environment
from planning_eval.ungrounded_environment import UngroundedFormatEnvironment
from planning_lab.algorithms.lats import lats
from planning_lab.algorithms.plan_and_solve import plan_and_solve
from planning_lab.algorithms.self_refine import reflect_and_refine
from planning_lab.algorithms.reflexion import reflexion
from planning_lab.algorithms.tree_of_thoughts import tree_of_thoughts


ARTIFACTS_DIR = (
    Path(__file__).resolve().parent / "artifacts"
)
ARTIFACTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# Groq pricing for llama-3.3-70b-versatile.
INPUT_PRICE_PER_1M = 0.59
OUTPUT_PRICE_PER_1M = 0.79


# ============================================================================
# Usage tracking
# ============================================================================

class UsageTracker:
    """
    Collect runtime LLM metrics.

    The planning implementations call the model internally, so this tracker
    wraps both normal invoke() and structured-output invoke().
    """

    def __init__(self) -> None:
        self.llm_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0

    def record(self, response: Any) -> None:
        self.llm_calls += 1

        usage = getattr(
            response,
            "usage_metadata",
            None,
        )

        if isinstance(usage, dict):
            self.input_tokens += int(
                usage.get("input_tokens", 0)
            )
            self.output_tokens += int(
                usage.get("output_tokens", 0)
            )
            self.total_tokens += int(
                usage.get("total_tokens", 0)
            )

    @property
    def cost_usd(self) -> float:
        cost = (
            self.input_tokens
            / 1_000_000
            * INPUT_PRICE_PER_1M
        ) + (
            self.output_tokens
            / 1_000_000
            * OUTPUT_PRICE_PER_1M
        )

        return round(cost, 6)


class TrackedRunnable:
    """Tracks an arbitrary LangChain Runnable returned by structured output."""

    def __init__(
        self,
        runnable: Any,
        tracker: UsageTracker,
    ) -> None:
        self._runnable = runnable
        self._tracker = tracker

    def invoke(
        self,
        input_data: Any,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        result = self._runnable.invoke(
            input_data,
            config=config,
            **kwargs,
        )

        self._tracker.record(result)

        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runnable, name)


def _invoke_with_retry(fn, *args, max_retries: int = 5, **kwargs):
    """Retry on Groq TPM rate limits with exponential backoff."""
    delay = 2.0
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except RateLimitError:
            if attempt == max_retries:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30.0)


class TrackedLLM:
    def __init__(self, llm: BaseChatModel, tracker: UsageTracker) -> None:
        self._llm = llm
        self._tracker = tracker
        # Function calling is supported by the small Groq model used for the
        # full benchmark. Unit-test fakes use their json_schema default.
        self.structured_output_method = "function_calling"

    def invoke(self, input_data, config=None, **kwargs):
        result = _invoke_with_retry(self._llm.invoke, input_data, config=config, **kwargs)
        self._tracker.record(result)
        return result

    def with_structured_output(self, *args, **kwargs):
        runnable = self._llm.with_structured_output(*args, **kwargs)
        return TrackedRunnable(runnable, self._tracker)


class TrackedRunnable:
    def __init__(self, runnable, tracker) -> None:
        self._runnable = runnable
        self._tracker = tracker

    def invoke(self, input_data, config=None, **kwargs):
        result = _invoke_with_retry(self._runnable.invoke, input_data, config=config, **kwargs)
        self._tracker.record(result)
        return result

    def __getattr__(self, name):
        return getattr(self._runnable, name)
# ============================================================================
# Model
# ============================================================================

def get_llm() -> ChatGroq:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Copy .env.example to .env and add a valid Groq API key."
        )

    return ChatGroq(
        api_key=api_key,
        model="llama-3.1-8b-instant",
        temperature=0,
        max_retries=2,
    )


# ============================================================================
# Serialization helpers
# ============================================================================

def _message_content(value: Any) -> str:
    content = getattr(
        value,
        "content",
        value,
    )

    if isinstance(content, str):
        return content.strip()

    return str(content).strip()


def _serialize(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()

    if isinstance(value, dict):
        return {
            str(key): _serialize(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _serialize(item)
            for item in value
        ]

    return value


def save_trace(trace: dict[str, Any]) -> Path:
    path = (
        ARTIFACTS_DIR
        / f"{trace['case_id']}__{trace['method']}.json"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            trace,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    return path


# ============================================================================
# Grounded evaluation
# ============================================================================

def evaluate_with_environment(
    case: TestCase,
    output: str,
    environment: Environment,
) -> dict[str, Any]:
    """
    The Environment is the source of truth for benchmark success.

    It must be the grounded MarketLoop implementation, not the old
    randomized toolkit evaluator.
    """
    feedback = environment.evaluate(output)

    grounded = bool(getattr(environment, "grounded", True))
    source_of_truth = getattr(
        environment,
        "source_of_truth",
        "MarketLoop SQLite database / MCP system of record.",
    )

    return {
        "success": feedback.success,
        "score": feedback.score,
        "details": feedback.details,
        "grounded": grounded,
        "ground_truth": _serialize(
            getattr(
                case,
                "ground_truth",
                None,
            )
        ),
        "source_of_truth": source_of_truth,
    }


# ============================================================================
# Decomposition-first
# ============================================================================

def run_decomposition_first(
    case: TestCase,
    llm: TrackedLLM,
    tracker: UsageTracker,
    environment: Environment,
) -> dict[str, Any]:

    started = time.perf_counter()

    plan = decompose_goal(
        goal=case.request,
        llm=llm,
    )

    executor = MarketLoopMCPExecutor(
        allow_mutations=False,
    )

    outputs = execute_plan(
        plan=plan,
        llm=llm,
        task_executor=executor,
        environment=environment,
    )

    final = final_output(
        plan=plan,
        outputs=outputs,
    )

    evaluation = evaluate_with_environment(
        case=case,
        output=final,
        environment=environment,
    )

    return {
        "case_id": case.id,
        "category": case.category,
        "method": "decomposition_first",
        "request": case.request,
        "success": evaluation["success"],
        "grounded": evaluation["grounded"],
        "source_of_truth": evaluation["source_of_truth"],
        "evaluation": evaluation,
        "plan": _serialize(plan),
        "node_outputs": _serialize(outputs),
        "final_output": final,
        "llm_calls": tracker.llm_calls,
        "input_tokens": tracker.input_tokens,
        "output_tokens": tracker.output_tokens,
        "total_tokens": tracker.total_tokens,
        "cost_usd": tracker.cost_usd,
        "latency_s": round(
            time.perf_counter() - started,
            4,
        ),
    }


# ============================================================================
# Dynamic decomposition
# ============================================================================

def run_dynamic_decomposition(
    case: TestCase,
    llm: TrackedLLM,
    tracker: UsageTracker,
    environment: Environment,
) -> dict[str, Any]:

    started = time.perf_counter()

    history = dynamic_decomposition(
        goal=case.request,
        llm=llm,
        max_steps=4,
    )

    output = "\n".join(
        f"{task}: {result}"
        for task, result in history
    )

    evaluation = evaluate_with_environment(
        case=case,
        output=output,
        environment=environment,
    )

    return {
        "case_id": case.id,
        "category": case.category,
        "method": "dynamic_decomposition",
        "request": case.request,
        "success": evaluation["success"],
        "grounded": evaluation["grounded"],
        "source_of_truth": evaluation["source_of_truth"],
        "evaluation": evaluation,
        "history": _serialize(history),
        "output": output,
        "llm_calls": tracker.llm_calls,
        "input_tokens": tracker.input_tokens,
        "output_tokens": tracker.output_tokens,
        "total_tokens": tracker.total_tokens,
        "cost_usd": tracker.cost_usd,
        "latency_s": round(
            time.perf_counter() - started,
            4,
        ),
    }


# ============================================================================
# Plan-and-Solve
# ============================================================================

def run_plan_and_solve(
    case: TestCase,
    llm: TrackedLLM,
    tracker: UsageTracker,
    environment: Environment,
) -> dict[str, Any]:

    started = time.perf_counter()

    output = plan_and_solve(
        question=case.request,
        llm=llm,
    )

    evaluation = evaluate_with_environment(
        case=case,
        output=output,
        environment=environment,
    )

    return {
        "case_id": case.id,
        "category": case.category,
        "method": "plan_and_solve",
        "request": case.request,
        "success": evaluation["success"],
        "grounded": evaluation["grounded"],
        "source_of_truth": evaluation["source_of_truth"],
        "evaluation": evaluation,
        "output": output,
        "llm_calls": tracker.llm_calls,
        "input_tokens": tracker.input_tokens,
        "output_tokens": tracker.output_tokens,
        "total_tokens": tracker.total_tokens,
        "cost_usd": tracker.cost_usd,
        "latency_s": round(
            time.perf_counter() - started,
            4,
        ),
    }


# ============================================================================
# Tree of Thoughts
# ============================================================================

def run_tree_of_thoughts(
    case: TestCase,
    llm: TrackedLLM,
    tracker: UsageTracker,
    environment: Environment,
) -> dict[str, Any]:

    started = time.perf_counter()

    thoughts = tree_of_thoughts(
        problem=case.request,
        llm=llm,
        depth=2,
        beam_width=2,
        search_strategy="bfs",
        prune_threshold=0.0,
    )

    candidates = [
        {
            "state": thought.state,
            "score": thought.score,
            "rationale": thought.rationale,
        }
        for thought in thoughts
    ]

    output = (
        thoughts[0].state
        if thoughts
        else ""
    )

    evaluation = evaluate_with_environment(
        case=case,
        output=output,
        environment=environment,
    )

    return {
        "case_id": case.id,
        "category": case.category,
        "method": "tree_of_thoughts",
        "request": case.request,
        "success": evaluation["success"],
        "grounded": evaluation["grounded"],
        "source_of_truth": evaluation["source_of_truth"],
        "evaluation": evaluation,
        "candidates": candidates,
        "output": output,
        "llm_calls": tracker.llm_calls,
        "input_tokens": tracker.input_tokens,
        "output_tokens": tracker.output_tokens,
        "total_tokens": tracker.total_tokens,
        "cost_usd": tracker.cost_usd,
        "latency_s": round(
            time.perf_counter() - started,
            4,
        ),
    }


# ============================================================================
# LATS
# ============================================================================

def run_lats(
    case: TestCase,
    llm: TrackedLLM,
    tracker: UsageTracker,
    environment: Environment,
) -> dict[str, Any]:

    started = time.perf_counter()

    result = lats(
        task=case.request,
        llm=llm,
        environment=environment,
        iterations=3,
        n_actions=2,
        exploration_weight=1.414,
    )

    evaluation = evaluate_with_environment(
        case=case,
        output=result.output,
        environment=environment,
    )

    return {
        "case_id": case.id,
        "category": case.category,
        "method": "lats",
        "request": case.request,
        "success": evaluation["success"],
        "grounded": evaluation["grounded"],
        "source_of_truth": evaluation["source_of_truth"],
        "evaluation": evaluation,
        "output": result.output,
        "best_score": result.best_score,
        "iterations": result.iterations,
        "tree": _serialize(result.root),
        "llm_calls": tracker.llm_calls,
        "input_tokens": tracker.input_tokens,
        "output_tokens": tracker.output_tokens,
        "total_tokens": tracker.total_tokens,
        "cost_usd": tracker.cost_usd,
        "latency_s": round(
            time.perf_counter() - started,
            4,
        ),
    }


def run_lats_ungrounded(
    case: TestCase,
    llm: TrackedLLM,
    tracker: UsageTracker,
    _grounded_environment: Environment,
) -> dict[str, Any]:
    """Run LATS with the required format-only baseline on the same case."""
    result = run_lats(
        case,
        llm,
        tracker,
        UngroundedFormatEnvironment(),
    )
    result["method"] = "lats_ungrounded"
    return result


# ============================================================================
# Self-Refine
# ============================================================================

def run_self_refine(
    case: TestCase,
    llm: TrackedLLM,
    tracker: UsageTracker,
    environment: Environment,
) -> dict[str, Any]:

    started = time.perf_counter()

    draft_response = llm.invoke(
        [
            (
                "system",
                "Produce the complete deliverable for the task.",
            ),
            (
                "human",
                f"""Task:
{case.request}

Return only the complete draft.""",
            ),
        ],
        temperature=0.2,
    )

    draft = _message_content(
        draft_response
    )

    refined = reflect_and_refine(
        goal=case.request,
        draft=draft,
        llm=llm,
        grounded_check=environment.evaluate,
        source_of_truth=environment.source_of_truth,
    )

    evaluation = evaluate_with_environment(
        case=case,
        output=refined.revised,
        environment=environment,
    )

    return {
        "case_id": case.id,
        "category": case.category,
        "method": "self_refine",
        "request": case.request,
        "success": evaluation["success"],
        "grounded": evaluation["grounded"],
        "source_of_truth": evaluation["source_of_truth"],
        "evaluation": evaluation,
        "draft": refined.draft,
        "critique": refined.critique,
        "revised": refined.revised,
        "grounded_issues": refined.grounded_issues,
        "llm_calls": tracker.llm_calls,
        "input_tokens": tracker.input_tokens,
        "output_tokens": tracker.output_tokens,
        "total_tokens": tracker.total_tokens,
        "cost_usd": tracker.cost_usd,
        "latency_s": round(
            time.perf_counter() - started,
            4,
        ),
    }


# ============================================================================
# Reflexion
# ============================================================================

def run_reflexion(
    case: TestCase,
    llm: TrackedLLM,
    tracker: UsageTracker,
    environment: Environment,
) -> dict[str, Any]:

    started = time.perf_counter()

    result = reflexion(
        task=case.request,
        llm=llm,
        environment=environment,
        max_trials=4,
        memory_size=3,
    )

    evaluation = evaluate_with_environment(
        case=case,
        output=result.output,
        environment=environment,
    )

    return {
        "case_id": case.id,
        "category": case.category,
        "method": "reflexion",
        "request": case.request,
        "success": evaluation["success"],
        "grounded": evaluation["grounded"],
        "source_of_truth": evaluation["source_of_truth"],
        "evaluation": evaluation,
        "output": result.output,
        "trials": [
            {
                "number": trial.number,
                "attempt": trial.attempt,
                "feedback": _serialize(
                    trial.feedback
                ),
                "reflection": trial.reflection,
            }
            for trial in result.trials
        ],
        "episodic_memory": result.memory,
        "llm_calls": tracker.llm_calls,
        "input_tokens": tracker.input_tokens,
        "output_tokens": tracker.output_tokens,
        "total_tokens": tracker.total_tokens,
        "cost_usd": tracker.cost_usd,
        "latency_s": round(
            time.perf_counter() - started,
            4,
        ),
    }


# ============================================================================
# One benchmark run
# ============================================================================

def run_one(
    case: TestCase,
    method_runner: Any,
) -> dict[str, Any]:

    base_llm = get_llm()
    tracker = UsageTracker()
    llm = TrackedLLM(
        base_llm,
        tracker,
    )

    # The benchmark must never use the reference toolkit's randomized
    # Environment. Every method receives a case-specific SQLite-backed
    # evaluator, and its feedback is recorded in the trace.
    environment = Environment(case)

    return method_runner(
        case,
        llm,
        tracker,
        environment,
    )


# ============================================================================
# Method applicability
# ============================================================================

def applicable_methods(
    category: str,
) -> list[tuple[str, Any]]:

    if category in {
        "DECOMP_FIRST",
        "DYNAMIC",
    }:
        return [
            (
                "decomposition_first",
                run_decomposition_first,
            ),
            (
                "dynamic_decomposition",
                run_dynamic_decomposition,
            ),
        ]

    if category == "LOOKAHEAD":
        return [
            (
                "plan_and_solve",
                run_plan_and_solve,
            ),
            (
                "tree_of_thoughts",
                run_tree_of_thoughts,
            ),
            (
                "lats_ungrounded",
                run_lats_ungrounded,
            ),
            (
                "lats",
                run_lats,
            ),
        ]

    if category == "REFLEXION":
        return [
            (
                "self_refine",
                run_self_refine,
            ),
            (
                "reflexion",
                run_reflexion,
            ),
        ]

    raise ValueError(
        f"Unsupported benchmark category: {category}"
    )


# ============================================================================
# Main
# ============================================================================

def run_all() -> list[dict[str, Any]]:
    all_results: list[dict[str, Any]] = []

    for case in TEST_CASES:
        print()
        print("=" * 90)
        print(
            f"{case.id} | "
            f"{case.category} | "
            f"{case.title}"
        )
        print("=" * 90)

        for method_name, runner in applicable_methods(
            case.category
        ):
            print(
                f"Running {method_name}..."
            )

            try:
                result = run_one(
                    case,
                    runner,
                )

                trace = {
                    "case_id": case.id,
                    "category": case.category,
                    "method": method_name,
                    "request": case.request,
                    **result,
                }

                # Keep the top-level identifiers deterministic.
                trace["method"] = method_name

                # Save trace.
                path = save_trace(
                    trace
                )

                all_results.append(
                    trace
                )

                print(
                    f"  success={trace['success']} "
                    f"calls={trace['llm_calls']} "
                    f"tokens={trace['total_tokens']} "
                    f"latency={trace['latency_s']}s "
                    f"cost=${trace['cost_usd']} "
                    f"-> {path.name}"
                )

            except Exception as exc:
                error_trace = {
                    "case_id": case.id,
                    "category": case.category,
                    "method": method_name,
                    "request": case.request,
                    "success": False,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                }

                path = save_trace(
                    error_trace
                )

                all_results.append(
                    error_trace
                )

                print(
                    f"  ERROR: {type(exc).__name__}: {exc}"
                )
                print(
                    f"  Trace saved to {path}"
                )

    summary_path = (
        ARTIFACTS_DIR
        / "benchmark_results.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            all_results,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    print()
    print("=" * 90)
    print(
        f"Completed {len(all_results)} benchmark runs."
    )
    print(
        f"Summary: {summary_path}"
    )
    print("=" * 90)

    return all_results


def rerun_failed() -> list[dict[str, Any]]:
    """Retry only failed benchmark records from the latest fixed-suite run."""
    summary_path = ARTIFACTS_DIR / "benchmark_results.json"
    with summary_path.open(encoding="utf-8") as file:
        previous_results = json.load(file)

    cases = {case.id: case for case in TEST_CASES}
    refreshed: list[dict[str, Any]] = []

    for previous in previous_results:
        if "error" not in previous:
            refreshed.append(previous)
            continue

        case = cases[previous["case_id"]]
        runner = dict(applicable_methods(case.category))[previous["method"]]
        print(f"Retrying {case.id}/{previous['method']}...")
        try:
            result = run_one(case, runner)
            trace = {
                "case_id": case.id,
                "category": case.category,
                "method": previous["method"],
                "request": case.request,
                **result,
            }
        except Exception as exc:
            trace = {
                "case_id": case.id,
                "category": case.category,
                "method": previous["method"],
                "request": case.request,
                "success": False,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }

        save_trace(trace)
        refreshed.append(trace)

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(refreshed, file, indent=2, ensure_ascii=False, default=str)
    return refreshed


if __name__ == "__main__":
    rerun_failed() if "--retry-failed" in sys.argv else run_all()
