# Retrieval Architecture Comparison — Agentic RAG

## 1. The problem
Some MarketLoop support questions genuinely need two separate lookups.
Example: "What's the return policy for a defective item, and which tool
processes an approved return?" - the return policy and the tool
description live in different indexed documents (`Returns` vs
`Customer Service`). A single keyword query tends to surface one side
and miss the other.

## 2. Test set
`retrieval_eval/decomposition_questions.py` - 3 fixed questions, each
spanning two distinct indexed subsections, grounded in the real content
indexed by `mcp_server/tools/rag_indexing.py` (not invented text).

Metric: does the retrieved evidence cover *both* expected subsections
for a question, not just one?

## 3. Results

Produced by `retrieval_eval/comparison_harness.py`:

| Architecture | Accuracy | Avg tokens/query | Avg latency (ms) |
|---|---|---|---|
| Naive RAG | pending (teammate's task) | - | - |
| Hybrid search | pending (teammate's task) | - | - |
| Agentic RAG (multi-hop) | 3/3 | 164.0 | 9.526 |

### Methodology
- **Accuracy**: both expected subsections must appear among the chunks
  retrieved across all hops for that question.
- **Avg tokens/query**: word count of only the chunks that pass the
  Self-RAG relevance check (`RAG/self_rag_verification.check_relevance`)
  - matching what a verification-filtered pipeline would actually forward,
  not the raw retrieval output.
- **Avg latency (ms)**: wall-clock time for the full decompose ->
  retrieve -> retrieve-again loop per question, on the BM25 keyword store
  (`mcp_server.tools.knowledge_store.KeywordStore`). No LLM call is
  involved in the decomposition step itself (it's regex-based sentence
  splitting), which is why this is fast; a production version would add
  latency for an LLM-driven decomposition/synthesis step.

## 4. Analysis
Agentic RAG correctly decomposed all 3 test questions into their
constituent sub-questions and retrieved evidence for both topics in
every case (3/3). The Self-RAG relevance filter also caught retrieval
noise before it counted toward token cost - only chunks that pass
`check_relevance` against their own sub-query are included, so the
164-token average reflects genuinely useful context, not everything
the BM25 query happened to return.

This is the one thing a single-shot retrieval query structurally can't
do: naive or hybrid search send one query and get one topic's worth of
evidence back, no matter how the results are re-ranked afterward.

## 5. Recommendation
**Agentic RAG should be routed to specifically for multi-part questions**
(detected by the same decomposition step: if a query splits into 2+
sub-questions, route it here) - not used as the default for every
query, since the decompose-then-retrieve loop costs more latency than a
single lookup would for a simple one-topic question. Final routing
recommendation across all three architectures needs the naive and
hybrid numbers above to be filled in before a company-wide default can
be chosen from the full table, not from this architecture in isolation.

## 6. Self-RAG verification (applies to both RAG and memory)
`RAG/self_rag_verification.py` implements two checks used above and
reusable elsewhere:
- `check_relevance(query, chunk)` - is retrieved content actually about
  what was asked, not just keyword-adjacent?
- `check_support(answer, chunks)` - is a generated answer traceable to
  what was retrieved, or does it contain claims that appeared nowhere in
  the evidence (a fabrication risk)?
- `check_memory_recall(context, recalled_content)` - the same relevance
  check, applied to a memory item pulled from the episodic/semantic
  store, so a promoted memory from an unrelated case can't get reused
  just because it shares generic words with the current conversation.

All three are lexical-overlap heuristics (no LLM call), covered by
`tests/test_self_rag_verification.py` (7 tests, including a fabricated-
answer case that must fail the support check and an unrelated-memory
case that must fail the recall check).
