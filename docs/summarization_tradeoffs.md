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
 
Produced by `context_eval/comparison_harness.py`:
 
| Strategy | Accuracy | Avg tokens/run | Avg latency (ms) |
|---|---|---|---|
| Sliding window (last 10 turns) | 0/12 | 45.0 | 0.009 |
| Observation masking (keep last 3 tool outputs) | 12/12 | 32.8 | 0.018 |
| Recursive summarization (every 10 turns) | 12/12 | 61.8 | 0.012 |
| Zone-based pruning | pending (teammate's task) | - | - |
 
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
  itself in Python (no LLM call involved in any of the three
  strategies - recursive summarization here is rule-based extraction of
  user-stated facts, not an LLM summarization call). This measures the
  strategy's own compute cost, not end-to-end response latency with an
  LLM in the loop - that's why the numbers are sub-millisecond.
## 4. Analysis
- **Sliding window fails completely (0/12)**: it truncates uniformly by
  position, so any detail stated more than 10 turns before the final
  question is gone - the exact shape of a real MarketLoop call.
- **Observation masking wins on tokens** (32.8 avg): it targets the
  actual bloat source (tool output), leaving dialogue - where the return
  reason lives - untouched.
- **Recursive summarization matches masking on accuracy** but costs
  ~90% more tokens, because it keeps the last 10 raw turns *in addition
  to* the summary, rather than aggressively dropping tool noise.
## 5. Recommendation (provisional, pending zone-based results)
**Observation masking** is the current leading candidate: it's the only
strategy that both preserves the return reason and does so at the lowest
token cost, because MarketLoop's bloat is tool-call noise, not dialogue
length. Final ranking will be confirmed once zone-based pruning numbers
are added to this table by the teammate covering that strategy.