# Context Window Management — Comparison
 
## 1. The problem
A MarketLoop support call to process a return can involve 30+ tool calls
(order lookup, shipment tracking, inventory checks, fee lookups) after the
customer states their return reason in the first message. Per MarketLoop's
return policy, the reason (damaged / wrong item vs. changed their mind)
decides whether a 15% restocking fee applies. If a context-pruning
strategy loses that detail, the agent risks charging a customer for
damage that wasn't their fault.
 
## 2. Test suite
`context_eval/scenario.py` builds 12 fixed transcripts: 4 return-reason
types x 3 variations each, ~35 randomized tool-call turns between the
reason and the final question "should a restocking fee apply?". The
suite is fixed once evaluation starts, per the lab's guardrail.
 
Metric: does the exact return reason survive in what's left of the
transcript after each strategy runs? (`context_eval/scenario.reason_survived`)
 
## 3. Results
 
Produced by `context_eval/comparison_harness.py`, all four required strategies:
 
| Strategy | Accuracy | Avg tokens/run | Avg latency (ms) |
|---|---|---|---|
| Sliding window (last 10 turns) | 0/12 | 45.0 | 0.007 |
| Observation masking (keep last 3 tool outputs) | 12/12 | 32.8 | 0.008 |
| Recursive summarization (every 10 turns) | 12/12 | 61.8 | 0.009 |
| Zone-based pruning (head=2, tail=10, middle=20%) | 12/12 | 80.8 | 0.009 |
 
### Methodology
- **Accuracy**: does the exact return reason string survive in the
  strategy's output for that test case? Binary per case, out of 12.
- **Avg tokens/run**: word count of the remaining transcript
  (`len(content.split())` summed across messages), not a model-specific
  tokenizer count. Used as a consistent, cheap proxy for context size
  across strategies - the *relative* ordering between strategies is what
  matters for the comparison, not the absolute token count a real LLM
  API would report.
- **Avg latency (ms)**: wall-clock time to run the pruning function
  itself in Python (no LLM call involved in any of the four
  strategies - recursive summarization here is rule-based extraction of
  user-stated facts, not an LLM summarization call). This measures the
  strategy's own compute cost, not end-to-end response latency with an
  LLM in the loop - that's why the numbers are sub-millisecond.
## 4. Analysis
- **Sliding window fails completely (0/12)**: it truncates uniformly by
  position, so any detail stated more than 10 turns before the final
  question is gone - the exact shape of a real MarketLoop call.
- **Observation masking wins on tokens** (32.8 avg) while keeping full
  accuracy: it targets the actual bloat source (tool output), leaving
  dialogue - where the return reason lives - untouched.
- **Recursive summarization matches masking on accuracy** but costs
  ~90% more tokens, because it keeps the last 10 raw turns *in addition
  to* the summary, rather than aggressively dropping tool noise.
- **Zone-based pruning also reaches full accuracy** (its fixed head
  window happens to always capture the return reason in this scenario,
  since it's stated in the first message), but it's the most expensive
  strategy here (80.8 avg tokens) - it keeps a head, a tail, *and* a
  sampled slice of the middle, which adds up even after dropping most of
  the tool noise.
## 5. Recommendation
**Observation masking** is the strategy MarketLoop ships with: it's the
only one that combines full accuracy with the lowest token cost, because
MarketLoop's bloat is tool-call noise, not dialogue length - masking
targets exactly that. Zone-based pruning is a reasonable fallback if a
future scenario buries the critical detail mid-conversation rather than
at the start (where a fixed head window wouldn't catch it), but for
MarketLoop's actual call shape today, masking is strictly better on both
metrics that matter.