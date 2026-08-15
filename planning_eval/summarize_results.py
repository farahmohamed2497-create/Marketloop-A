"""Create the Week 4 comparison table from real benchmark traces.

The script refuses to mark a benchmark complete while any applicable method
is missing or failed.  This prevents API errors and partial runs from being
presented as algorithm-quality results in the README.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from planning_eval.test_cases import TEST_CASES


ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
RESULTS_PATH = ARTIFACTS_DIR / "benchmark_results.json"
TABLE_PATH = ARTIFACTS_DIR / "comparison_table.md"

EXPECTED_METHODS = {
    "DECOMP_FIRST": {"decomposition_first", "dynamic_decomposition"},
    "DYNAMIC": {"decomposition_first", "dynamic_decomposition"},
    "LOOKAHEAD": {"plan_and_solve", "tree_of_thoughts", "lats_ungrounded", "lats"},
    "REFLEXION": {"self_refine", "reflexion"},
}


def load_results(path: Path = RESULTS_PATH) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        results = json.load(stream)
    if not isinstance(results, list):
        raise ValueError("benchmark_results.json must contain a JSON array")
    return results


def coverage_errors(results: list[dict[str, Any]]) -> list[str]:
    """Return missing-method and failed-run diagnostics for the fixed suite."""
    errors: list[str] = []
    by_case: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for record in results:
        case_id = record.get("case_id")
        category = record.get("category")
        method = record.get("method")
        if not all(isinstance(value, str) for value in (case_id, category, method)):
            errors.append("A benchmark record is missing case_id, category, or method.")
            continue
        by_case[(case_id, category)].append(record)

    expected_cases = {(case.id, case.category) for case in TEST_CASES}
    unexpected_cases = set(by_case) - expected_cases
    for case_id, category in sorted(unexpected_cases):
        errors.append(f"{case_id}: unsupported or unexpected benchmark case/category.")

    for case_id, category in sorted(expected_cases):
        records = by_case.get((case_id, category), [])
        required = EXPECTED_METHODS[category]
        seen = {record["method"] for record in records}
        missing = sorted(required - seen)
        if missing:
            errors.append(f"{case_id}: missing required method(s): {', '.join(missing)}.")
        for record in records:
            if record.get("method") in required and "error" in record:
                errors.append(
                    f"{case_id}/{record['method']}: run failed with "
                    f"{record['error'].get('type', 'unknown error')}.")
            elif record.get("method") in required and "total_tokens" not in record:
                errors.append(f"{case_id}/{record['method']}: metrics are missing.")

    return errors


def comparison_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, bool], list[dict[str, Any]]] = defaultdict(list)
    for record in results:
        if "error" not in record and "total_tokens" in record:
            grouped[(record["method"], bool(record.get("grounded", False)))].append(record)

    rows: list[dict[str, Any]] = []
    for (method, grounded), group in sorted(grouped.items()):
        rows.append(
            {
                "method": method,
                "grounded": grounded,
                "runs": len(group),
                "success_rate": sum(bool(item.get("success")) for item in group) / len(group),
                "avg_calls": mean(float(item["llm_calls"]) for item in group),
                "avg_tokens": mean(float(item["total_tokens"]) for item in group),
                "avg_latency_s": mean(float(item["latency_s"]) for item in group),
                "avg_cost_usd": mean(float(item["cost_usd"]) for item in group),
            }
        )
    return rows


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Week 4 Benchmark Comparison",
        "",
        "Generated from the fixed planning_eval suite. Do not edit values by hand.",
        "",
        "| Method | Grounded | Runs | Success rate | Avg. LLM calls | Avg. tokens | Avg. latency (s) | Avg. cost (USD) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {grounded} | {runs} | {success_rate:.1%} | {avg_calls:.2f} | "
            "{avg_tokens:.0f} | {avg_latency_s:.3f} | ${avg_cost_usd:.6f} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    results = load_results()
    errors = coverage_errors(results)
    if errors:
        print("Benchmark comparison was not generated because the run is incomplete:")
        for error in errors:
            print(f"- {error}")
        return 1

    table = render_markdown(comparison_rows(results))
    TABLE_PATH.write_text(table, encoding="utf-8")
    print(f"Wrote {TABLE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())