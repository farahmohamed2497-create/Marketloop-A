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
|---|---:|---:|---:|
| Sliding window (last 10) | 0/12 | 45.0 | 0.008 |
| Observation masking (keep last 3 tool outputs) | 12/12 | 32.8 | 0.014 |
| Recursive summarization (every 10 turns) | 12/12 | 61.8 | 0.011 |
| Zone-based pruning (head=2, tail=10, middle=20%) | 12/12 | 80.8 | 0.014 |

### Shipped context strategy: observation masking

MarketLoop ships observation/tool-output masking as the default. It retained the critical detail in every test case while using the fewest tokens of the strategies with full accuracy. This matches the real failure mode: tool output is the context bloat, not customer dialogue. Recursive summarization and zone pruning also retained the detail, but consumed more context without an accuracy gain on the fixed suite.

## Retrieval architecture

The retrieval corpus is the internal enterprise product catalog and policy material. Documents are chunked and embedded, stored in an ANN vector index, and accompanied by metadata payloads and a metadata index for filtering.

MarketLoop implements three retrieval paths:

1. **Naive RAG** (`RAG/naive_rag.py`): chunk, embed, retrieve, then generate from retrieved context.
2. **Hybrid search** (`RAG/hybrid_search.py`): combines vector similarity and BM25 keyword matching, which is important for exact policy codes, order identifiers, and product SKUs.
3. **Agentic RAG** (`RAG/agentic_rag.py`): decomposes a multi-part question, retrieves evidence per sub-question, observes coverage, and stops after the needed hops.

### Self-RAG-style verification

`RAG/self_rag_verification.py` checks two conditions before retrieved information is used:

- retrieved chunks must be relevant to the query;
- a proposed answer must be supported by the retrieved chunks.

The same relevance check is also available for recalled episodic and semantic memory, preventing an unrelated remembered event from being reused in the current support case.

### Retrieval evaluation

`retrieval_eval/` contains fixed question categories for general questions, identifier/citation-heavy questions, and multi-part decomposition questions. The final comparison runs every architecture against every question and reports accuracy, tokens per query, and latency per query. The results table below must be regenerated from `retrieval_eval/retrieval_comparison_harness.py` after all three pipelines are wired into the shared harness.

| Architecture | Accuracy | Avg. tokens/query | Avg. latency (ms) |
|---|---:|---:|---:|
| Naive RAG | Run harness | Run harness | Run harness |
| Hybrid search | Run harness | Run harness | Run harness |
| Agentic RAG (multi-hop) | Run harness | Run harness | Run harness |

MarketLoop uses hybrid retrieval for routine policy and exact-identifier questions, and routes decomposition-shaped questions to Agentic RAG when multiple retrieval rounds are warranted. The final measured table is the source of this decision.

## Demo and verification

The final demo should show:

1. a buffer overflow resulting in both a forget and an episodic promotion decision;
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
