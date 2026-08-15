# Week 4 Planning Demo Transcript

This transcript is backed by the fixed API-backed run in
`planning_eval/artifacts/benchmark_results.json` (50 records, no run errors).

## Dynamic divergence

Case B1 first queried the real sales-audit report for current low-stock items.
The trace records two low-stock products, Air Fryer (15) and Dell Laptop (20),
then added the conditional follow-up sales-activity investigation. This is the
branch that a fixed decomposition cannot know until the inventory observation
arrives.

## Search and grounding

Case C4's `lats_ungrounded` trace accepted a readable report-section ordering
with score 1.0 because the format-only baseline checks no database facts. The
paired grounded LATS trace used the SQLite-backed evaluator instead, which
requires the audit facts to tie out. This makes the source-of-truth difference
visible in the evidence rather than relying on model self-critique.

## Reflexion carries feedback across trials

Case D1's first Reflexion attempt invented sales figures and low-stock values.
The SQLite evaluator returned concrete failures, including missing Air Fryer,
missing Dell Laptop, and incorrect order/revenue values. Those reflections
entered the episodic buffer. The third attempt received score 1.0, giving a
real example of cross-trial correction.

## Measured outcome

`comparison_table.md` is generated from the same traces. Dynamic decomposition
reached 100% on its 10 applicable runs; decomposition-first reached 80%.
The table also records the cost-quality tradeoff for Plan-and-Solve,
Tree-of-Thoughts, LATS, Self-Refine, and Reflexion.
