"""Evaluation helpers for Person 2's ToT, Reflexion, and grounding work."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Callable

from planning_lab.algorithms.reflexion import reflexion
from planning_lab.algorithms.tree_of_thoughts import tree_of_thoughts
from planning_lab.models import EnvironmentFeedback

from .sales_audit_cases import SalesAuditCase


@dataclass
class CallMetrics:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if hasattr(value, "content"):
        return _text(value.content)
    if hasattr(value, "model_dump_json"):
        return value.model_dump_json()
    return json.dumps(value, default=str, sort_keys=True)


def _usage(value: Any, prompt: str) -> tuple[int, int]:
    usage = getattr(value, "usage_metadata", None) or {}
    if not usage:
        metadata = getattr(value, "response_metadata", None) or {}
        usage = metadata.get("token_usage", {})

    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))

    return (
        int(input_tokens) if input_tokens is not None else len(prompt.split()),
        int(output_tokens) if output_tokens is not None else len(_text(value).split()),
    )


class InstrumentedLLM:
    """Transparent wrapper that records normal and structured LLM calls."""

    def __init__(self, delegate: Any, metrics: CallMetrics | None = None) -> None:
        self.delegate = delegate
        self.metrics = metrics or CallMetrics()

    def _record(self, prompt: Any, call: Callable[[], Any]) -> Any:
        prompt_text = _text(prompt)
        started = perf_counter()
        result = call()

        self.metrics.calls += 1
        self.metrics.latency_ms += (perf_counter() - started) * 1000

        input_tokens, output_tokens = _usage(result, prompt_text)
        self.metrics.input_tokens += input_tokens
        self.metrics.output_tokens += output_tokens

        return result

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        return self._record(messages, lambda: self.delegate.invoke(messages, **kwargs))

    def with_structured_output(self, *args: Any, **kwargs: Any) -> Any:
        runner = self.delegate.with_structured_output(*args, **kwargs)
        wrapper = self

        class InstrumentedRunner:
            def invoke(self, messages: Any, **invoke_kwargs: Any) -> Any:
                return wrapper._record(
                    messages,
                    lambda: runner.invoke(messages, **invoke_kwargs),
                )

        return InstrumentedRunner()


class UngroundedActionEnvironment:
    """Weak baseline used only to compare grounded and ungrounded validation."""

    def __init__(self, required_fragment: str) -> None:
        self.required_fragment = required_fragment.lower()

    def evaluate(self, state: str) -> EnvironmentFeedback:
        accepted = self.required_fragment in state.lower()

        return EnvironmentFeedback(
            success=accepted,
            score=1.0 if accepted else 0.0,
            details=(
                ["Format-only baseline accepted the requested wording."]
                if accepted
                else [f"Output must mention {self.required_fragment!r}."]
            ),
        )


def _result(
    case: SalesAuditCase,
    method: str,
    output: str,
    success: bool,
    grounded: bool,
    metrics: CallMetrics,
) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "method": method,
        "success": success,
        "grounded": grounded,
        "output": output,
        "metrics": {**asdict(metrics), "total_tokens": metrics.total_tokens},
    }


def evaluate_tot_case(
    case: SalesAuditCase,
    llm: Any,
    *,
    search_strategy: str = "bfs",
    prune_threshold: float = 0.5,
) -> dict[str, Any]:
    instrumented = InstrumentedLLM(llm)

    thoughts = tree_of_thoughts(
        case.prompt,
        instrumented,
        search_strategy=search_strategy,
        prune_threshold=prune_threshold,
    )

    output = thoughts[0].state if thoughts else ""

    return _result(
        case,
        f"tot-{search_strategy}",
        output,
        case.expected_action_fragment.lower() in output.lower(),
        False,
        instrumented.metrics,
    )


def evaluate_reflexion_case(
    case: SalesAuditCase,
    llm: Any,
    environment: Any,
    *,
    grounded: bool,
) -> dict[str, Any]:
    instrumented = InstrumentedLLM(llm)
    outcome = reflexion(case.prompt, instrumented, environment)

    return _result(
        case,
        "reflexion",
        outcome.output,
        outcome.success,
        grounded,
        instrumented.metrics,
    )


def run_person2_benchmark(
    cases: tuple[SalesAuditCase, ...],
    llm: Any,
    grounded_environment_factory: Callable[[SalesAuditCase], Any],
) -> list[dict[str, Any]]:
    """Run fixed Person 2 cases and record ToT/Reflexion metrics."""

    records: list[dict[str, Any]] = []

    for case in cases:
        records.append(evaluate_tot_case(case, llm))

        records.append(
            evaluate_reflexion_case(
                case,
                llm,
                UngroundedActionEnvironment(case.expected_action_fragment),
                grounded=False,
            )
        )

        records.append(
            evaluate_reflexion_case(
                case,
                llm,
                grounded_environment_factory(case),
                grounded=True,
            )
        )

    return records


def summarize_person2_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate actual measured records without inventing results."""

    grouped: dict[tuple[str, bool], list[dict[str, Any]]] = {}

    for record in records:
        grouped.setdefault((record["method"], record["grounded"]), []).append(record)

    summary: list[dict[str, Any]] = []

    for (method, grounded), group in sorted(grouped.items()):
        count = len(group)

        summary.append(
            {
                "method": method,
                "grounded": grounded,
                "cases": count,
                "success_rate": sum(item["success"] for item in group) / count,
                "avg_calls": sum(item["metrics"]["calls"] for item in group) / count,
                "avg_total_tokens": sum(
                    item["metrics"]["total_tokens"] for item in group
                )
                / count,
                "avg_latency_ms": sum(
                    item["metrics"]["latency_ms"] for item in group
                )
                / count,
            }
        )

    return summary