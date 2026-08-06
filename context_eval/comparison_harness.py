"""
Runs every implemented context-management strategy against the fixed
return-reason test suite and produces the comparison table required by
the lab (accuracy, tokens, latency -> final choice, justified by the
table).
 
All four strategies (sliding window, masking, summarization,
zone-based pruning) are included below.
"""
 
from __future__ import annotations
 
import sys
import time
from pathlib import Path
 
sys.path.append(str(Path(__file__).resolve().parents[1]))
 
from context_eval.scenario import build_test_suite, reason_survived
from mcp_server.memory.sliding_window import apply_sliding_window
from mcp_server.memory.masking import mask_tool_outputs
from mcp_server.memory.summarization import RecursiveSummarizer
from mcp_server.memory.zone_pruning import zone_prune
 
 
def count_tokens(messages) -> int:
    return sum(len(m.get("content", "").split()) for m in messages)
 
 
def run_strategy(name: str, fn, cases) -> dict:
    correct = 0
    total_tokens = 0
    start = time.perf_counter()
 
    for case in cases:
        result = fn(case["messages"])
        total_tokens += count_tokens(result)
        if reason_survived(result, case["reason"]):
            correct += 1
 
    elapsed = time.perf_counter() - start
    n = len(cases)
 
    return {
        "strategy": name,
        "accuracy": f"{correct}/{n}",
        "accuracy_pct": round(100 * correct / n, 1),
        "avg_tokens": round(total_tokens / n, 1),
        "avg_latency_ms": round(1000 * elapsed / n, 3),
    }
 
 
def main():
    cases = build_test_suite()
 
    strategies = [
        ("Sliding window (last 10)", lambda m: apply_sliding_window(m, window_size=10)),
        ("Observation masking (keep last 3 tool outputs)", lambda m: mask_tool_outputs(m, keep_last_outputs=3)),
        ("Recursive summarization (every 10 turns)", lambda m: RecursiveSummarizer(threshold=10).summarize(m)),
        ("Zone-based pruning (head=2, tail=10, middle=20%)", lambda m: zone_prune(m, head_size=2, tail_size=10, middle_keep_ratio=0.2)),
    ]
 
    results = [run_strategy(name, fn, cases) for name, fn in strategies]
 
    # Print as a markdown table
    print(f"\nTest suite: {len(cases)} cases (return-reason recall, ~35 tool-noise turns each)\n")
    header = "| Strategy | Accuracy | Avg tokens/run | Avg latency (ms) |"
    sep = "|---|---|---|---|"
    print(header)
    print(sep)
    for r in results:
        print(f"| {r['strategy']} | {r['accuracy']} | {r['avg_tokens']} | {r['avg_latency_ms']} |")
 
    return results
 
 
if __name__ == "__main__":
    main()