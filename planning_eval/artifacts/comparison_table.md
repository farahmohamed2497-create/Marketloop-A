# Week 4 Benchmark Comparison

Generated from the fixed planning_eval suite. Do not edit values by hand.

| Method | Grounded | Runs | Success rate | Avg. LLM calls | Avg. tokens | Avg. latency (s) | Avg. cost (USD) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decomposition_first | True | 10 | 80.0% | 5.40 | 4459 | 38.875 | $0.003009 |
| dynamic_decomposition | True | 10 | 100.0% | 4.30 | 214 | 47.842 | $0.000137 |
| lats | True | 5 | 0.0% | 10.40 | 3186 | 93.059 | $0.002253 |
| lats_ungrounded | False | 5 | 20.0% | 13.00 | 2842 | 81.268 | $0.002063 |
| plan_and_solve | True | 5 | 0.0% | 2.00 | 745 | 10.307 | $0.000547 |
| reflexion | True | 5 | 40.0% | 6.80 | 8139 | 79.522 | $0.005488 |
| self_refine | True | 5 | 40.0% | 3.00 | 3860 | 37.844 | $0.002595 |
| tree_of_thoughts | True | 5 | 0.0% | 8.60 | 0 | 37.128 | $0.000000 |
