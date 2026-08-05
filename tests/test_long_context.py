"""
Replaces the old generic-fact tests. Grounded in the real MarketLoop
scenario: does each strategy preserve the customer's return reason
across a long, tool-heavy support transcript? See context_eval/scenario.py
for the fixed test suite these draw from.
"""
 
import sys
from pathlib import Path
 
sys.path.append(str(Path(__file__).resolve().parents[1]))
 
from context_eval.scenario import build_case, reason_survived
from mcp_server.memory.sliding_window import apply_sliding_window
from mcp_server.memory.masking import mask_tool_outputs
from mcp_server.memory.summarization import RecursiveSummarizer
 
 
CASE = build_case(
    reason="item arrived damaged in shipping",
    no_fee_expected=True,
    noise_turns=35,
    seed=1,
)
 
 
def test_sliding_window_loses_reason_on_long_transcript():
    """Known, real failure mode: sliding window keeps only the last 10
    turns, so a reason stated in turn 1 of a 35+ turn call is dropped."""
    result = apply_sliding_window(CASE["messages"], window_size=10)
    assert reason_survived(result, CASE["reason"]) is False
 
 
def test_masking_preserves_reason():
    result = mask_tool_outputs(CASE["messages"], keep_last_outputs=3)
    assert reason_survived(result, CASE["reason"]) is True
 
 
def test_summarization_preserves_reason():
    result = RecursiveSummarizer(threshold=10).summarize(CASE["messages"])
    assert reason_survived(result, CASE["reason"]) is True