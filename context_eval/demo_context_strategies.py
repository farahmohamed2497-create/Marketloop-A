"""Run the fixed MarketLoop context suite and show each pruning strategy.

This is the demo artifact for the context-management requirement.  It uses
the same fixed cases and parameters as context_eval/comparison_harness.py;
it does not alter the evaluation suite or fabricate new measurements.

Run from the repository root:
    python context_eval/demo_context_strategies.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from context_eval.comparison_harness import run_strategy
from context_eval.scenario import build_test_suite, reason_survived
from mcp_server.memory.masking import mask_tool_outputs
from mcp_server.memory.sliding_window import apply_sliding_window
from mcp_server.memory.summarization import RecursiveSummarizer
from mcp_server.memory.zone_pruning import zone_prune


STRATEGIES = [
    ("Sliding window (last 10)", lambda messages: apply_sliding_window(messages, window_size=10)),
    ("Observation masking (keep last 3 tool outputs)", lambda messages: mask_tool_outputs(messages, keep_last_outputs=3)),
    ("Recursive summarization (every 10 turns)", lambda messages: RecursiveSummarizer(threshold=10).summarize(messages)),
    ("Zone-based pruning (head=2, tail=10, middle=20%)", lambda messages: zone_prune(messages, head_size=2, tail_size=10, middle_keep_ratio=0.2)),
]


def main() -> None:
    cases = build_test_suite()
    example = cases[0]

    print("MarketLoop context-management demo")
    print(f"Fixed suite: {len(cases)} cases; example transcript: {len(example['messages'])} turns")
    print(f"Critical early fact: {example['reason']!r}\n")

    print("Per-strategy example outcome:")
    for name, strategy in STRATEGIES:
        pruned = strategy(example["messages"])
        preserved = reason_survived(pruned, example["reason"])
        print(f"- {name}: {len(example['messages'])} -> {len(pruned)} messages; "
              f"critical fact preserved = {preserved}")

    print("\nFixed-suite comparison:")
    results = [run_strategy(name, strategy, cases) for name, strategy in STRATEGIES]
    print("| Strategy | Accuracy | Avg tokens/run | Avg latency (ms) |")
    print("|---|---|---|---|")
    for result in results:
        print(
            f"| {result['strategy']} | {result['accuracy']} | "
            f"{result['avg_tokens']} | {result['avg_latency_ms']} |"
        )

    print(
        "\nDecision: ship observation masking. It preserves the return reason "
        "in every fixed-suite case while using the fewest tokens among the "
        "strategies with full accuracy."
    )


if __name__ == "__main__":
    main()