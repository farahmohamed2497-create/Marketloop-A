# MarketLoop State Graphs

This package implements three stateful workflows for the MarketLoop
final project.

## Graph 1 - Return / Refund Resolution

Handles returns that may require:

- return eligibility checks
- policy grounding
- human approval
- waiting for physical return
- inspection
- refund processing

LLM additions:

1. Task Decomposition
2. RAG / Policy Grounding

The graph can pause while waiting for the return and can pause for
human approval when a return exceeds the configured amount threshold.

---

## Graph 2 - Delayed Order Escalation

Handles delayed orders that require:

- order analysis
- candidate action generation
- constrained tool actions
- external carrier response
- escalation
- human approval

LLM additions:

1. LATS / Tree of Thoughts
2. Constrained ReAct

The graph waits for an external carrier response and can return to
the waiting state after an approved retry.

---

## Graph 3 - Inventory Discrepancy Recovery

Handles inventory discrepancies between system records and warehouse
counts.

LLM additions:

1. RAG / Inventory Policy Grounding
2. Constrained ReAct

The workflow requires warehouse confirmation and human approval before
changing the system-of-record inventory quantity.

---

## Checkpointing

A checkpoint is persisted after each meaningful state transition.

The checkpoint contains:

- run ID
- graph name
- current node
- graph status
- state data
- version
- timestamp

A new process can restore the latest checkpoint using
`RecoveryService`.

---

## HITL versus Failure Tickets

HITL is an expected pause.

Examples:

- amount above threshold
- low confidence
- policy conflict
- irreversible action

A failure ticket is different.

A ticket is created when an unexpected failure occurs, such as:

- tool error
- invalid data
- schema validation failure
- unhandled node exception

HITL:

    running -> paused_hitl -> admin decision -> resume

Failure:

    running -> failed -> ticket -> resolved -> resume

---

## Running the package

From the repository root:

```bash
python -m state_graph