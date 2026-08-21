# MarketLoop: Memory and Grounded Retrieval for MCP Support Agents

MarketLoop is an MCP-based support assistant for an enterprise marketplace. It already exposes scoped tools for orders, returns, inventory, customer-service records, and reports. This extension gives that agent persistent memory and document-grounded retrieval without rebuilding the existing MCP server or SQLite database.

## The problem

MarketLoop return-support calls can exceed 35 turns. Most of that context is noisy tool output: order lookups, shipment tracking, inventory checks, and fee checks. A customer may state the important fact—such as *"the item arrived damaged"*—only once, near the start of the call. That fact determines whether the 15% restocking fee applies. Losing it can cause an incorrect charge.

The same agents also need answers that do not belong in a database tool call, including return-policy, shipping-SLA, warranty, and product-catalog questions. Those answers must be grounded in retrieved internal documents, especially when a question contains an exact policy identifier or needs multiple policy sections.

## Project structure

```text
agent/                 Agent client and demos
mcp_server/            Existing MCP server, tools, database access, and memory
  memory/              Short-term, episodic, and semantic memory components
context_eval/          Fixed long-context suite and four-strategy comparison
RAG/                   Vector store, retrieval pipelines, and verification
retrieval_eval/        Fixed retrieval questions and evaluation harness
db/                    Existing SQLite schema and seed data
tests/                 Unit and integration tests
```

## Setup and run

```bash
python -m pip install -r requirements.txt
python db/init_db.py
python context_eval/comparison_harness.py
python retrieval_eval/retrieval_comparison_harness.py
python -m pytest -q
```

Do not commit API keys, embedding credentials, or generated vector-database artifacts. Store secrets in `.env`, which must remain ignored by Git.

### API setup

Copy `.env.example` to `.env` and add `GROQ_API_KEY` to run the API-backed
planning benchmark and Groq rag paths. `MISTRAL_API_KEY` is needed only for
`planning_lab.cli`. Never commit `.env`.

```bash
python -m planning_eval.run_benchmark
python -m planning_eval.summarize_results
```

## Memory architecture

### Short-term memory and scratchpad

`mcp_server/memory/rolling_buffer.py` holds the rolling transcript used for the current conversation. It is deliberately separate from `mcp_server/memory/scratchpad.py`, which stores the agent's current plan, sub-goal, and working state. Context pruning never removes the scratchpad.

When the rolling buffer overflows, `mcp_server/memory/promote_drop_router.py` makes and logs a visible decision for each aging item:

- **Forget:** transient greetings, duplicate tool output, or facts with no future support value.
- **Promote to episodic:** customer-specific events that can affect a later support interaction.

The router never writes directly to semantic memory.

### Episodic and semantic memory

Episodic events are stored in `mcp_server/memory/episodic_store.py`. Semantic facts are created only by the separate periodic consolidation process in `mcp_server/memory/consolidation.py`, then stored in `mcp_server/memory/semantic_memory.py`.

Consolidation is responsible for updating facts, retaining version history, applying expiration dates, and recording how contradictory episodes were resolved. The conflict demo in `mcp_server/memory/demo_conflict.py` provides a reproducible example.

## Context-window evaluation

The fixed suite in `context_eval/scenario.py` contains 12 return-support cases. Each buries an early return reason beneath approximately 35 tool-heavy turns, then asks for that reason at the end. `context_eval/comparison_harness.py` runs the same suite for all four strategies.

| Strategy | Accuracy | Avg. tokens/run | Avg. latency (ms) |
| Sliding window (last 10) | 0/12 | 45.0 | 0.008 |
| Observation masking (keep last 3 tool outputs) | 12/12 | 32.8 | 0.014 |
| Recursive summarization (every 10 turns) | 12/12 | 61.8 | 0.011 |
| Zone-based pruning (head=2, tail=10, middle=20%) | 12/12 | 80.8 | 0.014 |

### Shipped context strategy: observation masking

MarketLoop ships observation/tool-output masking as the default. It retained the critical detail in every test case while using the fewest tokens of the strategies with full accuracy. This matches the real failure mode: tool output is the context bloat, not customer dialogue. Recursive summarization and zone pruning also retained the detail, but consumed more context without an accuracy gain on the fixed suite.

## Retrieval architecture

The retrieval corpus is the internal enterprise product catalog and policy material. Documents are chunked and embedded, stored in an ANN vector index, and accompanied by metadata payloads and a metadata index for filtering.

MarketLoop implements three retrieval paths:

1. **Naive RAG** (`rag/naive_rag.py`): chunk, embed, retrieve, then generate from retrieved context.
2. **Hybrid search** (`rag/hybrid_search.py`): combines vector similarity and BM25 keyword matching, which is important for exact policy codes, order identifiers, and product SKUs.
3. **Agentic RAG** (`rag/agentic_rag.py`): decomposes a multi-part question, retrieves evidence per sub-question, observes coverage, and stops after the needed hops.

### Self-rag-style verification

`RAG/self_rag_verification.py` checks two conditions before retrieved information is used:

- retrieved chunks must be relevant to the query;
- a proposed answer must be supported by the retrieved chunks.

The same relevance check is also available for recalled episodic and semantic memory, preventing an unrelated remembered event from being reused in the current support case.

### Retrieval evaluation

`retrieval_eval/` contains fixed question categories for general questions, identifier/citation-heavy questions, and multi-part decomposition questions. The shared harness runs every architecture against the same nine fixed questions and reports evidence-check accuracy, tokens per query, and latency per query.

| Architecture | Accuracy | Avg. tokens/query | Avg. latency (ms) |
| Naive RAG (vector only) | 7/9 | 1277.8 | 9.823 |
| Hybrid Search (vector + BM25) | 9/9 | 1353.8 | 8.360 |
| Agentic RAG (Hybrid multi-hop) | 9/9 | 1652.0 | 9.979 |

Hybrid Search is the shipping default: it reached full evidence coverage with the lowest measured latency and lower token use than Agentic RAG. Agentic RAG is reserved for multi-part questions, where it decomposes the request and runs Hybrid Search for each sub-question. Naive RAG remains as the vector-only baseline for the evaluation.

## Demo and verification

The final demo should show:

1. a buffer overflow resulting in both a forged and an episodic promotion decision;
2. periodic consolidation resolving a real contradictory fact while retaining history;
3. all four context strategies running against the fixed suite;
4. Naive, Hybrid, and Agentic retrieval answering the same question set;
5. a Self-RAG check passing supported evidence and rejecting unsupported or irrelevant evidence;
6. an end-to-end agent request that uses both memory and retrieval.

## Key files

- `agent/client.py` — live agent loop.
- `mcp_server/server.py` — MCP server and existing tool registration.
- `agent/memory_demo.py` — memory demo path.
- `context_eval/comparison_harness.py` — context comparison table.
- `retrieval_eval/retrieval_comparison_harness.py` — retrieval comparison table.
- `RAG/self_rag_verification.py` — retrieval and memory verification checks.

## Note on git history

Some commits appear under more than one identity for the same
contributor (e.g. `farahmohamed2497-create` and `Farah Mohamed`;
`youssef` and `youssef mahmoud`) due to unset `git config user.name`
on different machines. Commit volume per name does not reflect actual
ownership distribution — see issue rationale and PR descriptions for
per-concern ownership.

## Integrated Planning Workflow

The planning pipeline integrates decomposition, task execution,
self-refinement, and evaluation.

### DAG Workflow

The default DAG mode follows:

Goal
→ Decomposition
→ DAG Execution
→ Final Synthesis
→ Self-Refine
→ Evaluation
→ Final Result

### Decomposition

The goal is converted into a validated directed acyclic graph (DAG).
Each task has a unique identifier, an instruction, and optional
dependencies.

Independent tasks can execute in parallel, while dependent tasks
wait for prerequisite outputs.

### Self-Refine

After the terminal synthesis task produces a draft, Self-Refine can:

1. Run deterministic checks.
2. Generate an independent critique.
3. Revise the draft using the detected issues.

If no issues are found, the existing draft is preserved.

### Evaluation

The evaluation layer provides metrics for:

- Decomposition
- Plan-and-Solve
- Self-Refine

Each evaluation reports:

- total checks
- passed checks
- failed checks
- pass rate

Evaluation results are stored in the generated run artifact.

### Run Artifacts

Each execution stores its results under the `artifacts/` directory.
Artifacts can contain the generated plan, task outputs, final result,
reflection information, and evaluation metrics.

## Week 4 Sales Audit Planning Extension

The separate `SalesAuditPlanningAgent` uses the existing MarketLoop MCP tools
and SQLite database to produce multi-step sales audits. It leaves the existing
memory/RAG agent path intact. `planning_lab/` is adapted from the required
reference planning toolkit; `planning_eval/` contains the fixed 20-case suite,
the API-backed runner, and JSON traces.

- Decomposition-first constructs and validates one complete DAG before
  execution; dynamic decomposition observes each result and can change the
  next step.
- Plan-and-Solve, Tree-of-Thoughts, and LATS are available through the routing
  layer for fixed synthesis, candidate comparison, and validated actions.
- Self-Refine uses a draft/critique/revision cycle; Reflexion carries a capped
  memory of grounded failures into later trials.
- `CaseGroundedEnvironment` validates candidate claims against the real
  MarketLoop SQLite tables. The `lats_ungrounded` baseline intentionally
  checks only answer shape, not facts.

### Fixed-suite comparison

The following table was generated from `planning_eval/artifacts/benchmark_results.json`
by `python -m planning_eval.summarize_results`.

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

The production default for top-level sales-audit decomposition is dynamic
decomposition: it achieved 100% success on the fixed suite, versus 80% for
decomposition-first. For correction, Self-Refine is cheaper for a single
revision, while Reflexion remains available where cross-trial learning is
needed. LATS is retained for constrained, externally validated restock-action
subtasks; the current benchmark makes its high cost and poor factual success
visible rather than hiding it. The ungrounded LATS baseline accepted one
format-shaped but unverified output, demonstrating why format-only critique is
not a shipping decision signal.

## State Graphs — Graph 1 Refund Workflow

### Refund Graph

Graph 1 models a refund request as a persistent state graph rather than a linear
function call. The graph keeps its execution state in `GraphState` and persists
checkpoints after meaningful transitions.

The current flow is:

```text
awaiting_input
      |
      v
    lats
      |
      +--------------------+
      |                    |
      v                    v
evaluate_refund         HITL pause
                           |
                           v
                      admin decision
                           |
                           v
                        resume


                                                

                                                                        LATS decision step

The LATS node explores multiple refund candidates and stores the selected result
inside GraphState.data["lats"].

The stored fields include:

success
output
best_score
iterations

This state is preserved so later validation and human review can inspect the
decision that led to the current graph state.

HITL trigger conditions

A refund request is escalated to a human administrator when at least one of the
following conditions is true:

LATS confidence score is below 0.70.
Refund amount is greater than 500.
The proposed action violates the configured refund policy.

These conditions are implemented in:

state_graph/hitl/policy.py

HITL pause and persistence

When a HITL condition is triggered, HITLNode.pause():

Creates a unique waiting_request_id.
Creates a paused state with status waiting.
Persists the full state snapshot with the HITL request.
Stores the request as pending.
Returns a TransitionResult with status="waiting".

The persisted HITL request contains:

run ID
graph name
reason for escalation
complete serialized graph state
request status
administrator decision
creation and resolution timestamps

The graph therefore stops without losing the state collected before the
escalation.

HITL resume flow

The intended resume flow is:

Graph running
    |
    v
HITL condition triggered
    |
    v
Create HITL request
    |
    v
Persist full state
    |
    v
Graph status = waiting
    |
    v
Administrator reviews request
    |
    +------ approve ------+
    |                     |
    +------ reject -------+
                          |
                          v
                    store decision
                          |
                          v
                  resume graph run
                          |
                          v
                continue from checkpoint

                The administrator decision is part of the persisted HITL request and is used
when the run resumes. The run is not restarted from the beginning.

Failure vs HITL

HITL and failure recovery represent different paths:

HITL is an expected pause caused by a decision that the agent is not
authorized to make autonomously.
Failure recovery is an unplanned interruption such as a tool error,
validation error, or unusable model output.

HITL requests use the waiting state and a waiting_request_id, while failure
recovery uses the failure ticket path and waiting_ticket_id.

Current implementation status

Implemented:

LATS integration into Graph 1.
HITL threshold, policy, and confidence conditions.
Persistent HITL request creation.
Full-state persistence during HITL pause.
Automated tests for HITL triggering and pause behavior.

Pending:

HTML admin platform and task-queue integration.
Final end-to-end resume demonstration through the admin UI.
